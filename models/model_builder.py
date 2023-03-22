import copy, random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
from torch.distributions import RelaxedBernoulli
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from models.optimizers import Optimizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import LongformerModel, BertModel, T5ForConditionalGeneration, BartForConditionalGeneration
from models.encoder import SentenceClassification, ClusterClassification, TransformerEncoder, Classifier, PairClassification
from models.slot_attn import SlotAttention, SoftKMeans
from models.spectral_clustering import SpectralCluser
from sentence_transformers.models import Pooling

torch.autograd.set_detect_anomaly(True)

def build_optim(args, model, checkpoint, lr=None, warmup_steps=None):
    """ Build optimizer """
    if (checkpoint is not None) and (args.reset_optimizer == False):
        for key in checkpoint:
            print (key)
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

        if lr is not None:
            optim.learning_rate = lr

    else:
        if lr is None:
            lr = args.lr
        if warmup_steps is None:
            warmup_steps = args.warmup_steps
        optim = Optimizer(
            args.optim, lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=warmup_steps)

    params = []
    for name, para in model.named_parameters():
        if para.requires_grad:
            params.append((name, para))
    optim.set_parameters(params)
    return optim


def build_optim_encdec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

        optim.learning_rate = args.lr_encdec

    else:
        optim = Optimizer(
            args.optim, args.lr_encdec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_encdec)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('abs_model')]
    optim.set_parameters(params)

    return optim


def build_optim_planner(args, model, checkpoint):
    """ Build optimizer """

    optim = None
    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            #raise RuntimeError(
            #    "Error: loaded Adam optimizer from existing model" +
            #    " but optimizer state is empty")
            optim = None

    if optim is None:
        optim = Optimizer(
            args.optim, args.lr_planner, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_planner)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('abs_model')]
    optim.set_parameters(params)

    return optim


def get_generator(vocab_size, dec_hidden_size, device, pre_trained_generator=None):
    if pre_trained_generator is not None:
        gen = pre_trained_generator
    else:
        gen = nn.Linear(dec_hidden_size, vocab_size)

    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        gen,
        gen_func
    )

    generator.to(device)
    return generator



class AbsSummarizerNewVersion(nn.Module):
    def __init__(self, args, device, vocab_size, checkpoint=None):
        super(AbsSummarizerNewVersion, self).__init__()
        self.args = args
        self.device = device

        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name)

        self.vocab_size = vocab_size
        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        print (self.vocab_size, len(self.original_tokenizer))
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)
        self.pad_id = self.original_tokenizer.pad_token_id

        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, 
                                        self.model.config.hidden_size, 
                                        device, 
                                        pre_trained_generator=self.model.lm_head)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            print ('Initialize parameters for generator...')

        self.to(device)


    def _log_generation_stats(self, scores, target, src):
        pred = scores.max(2)[1]
        non_padding = target.ne(self.pad_id)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        log_info = {'num_non_padding':num_non_padding, 'num_correct':num_correct, 'n_docs':int(src.size(0))}
        return log_info


    def forward(self, src, tgt, mask_src, mask_tgt, run_decoder=True):

        labels = tgt[:,1:].contiguous()
        labels[labels == self.pad_id] = -100

        if not run_decoder:
            output_sequences = self.model.generate(input_ids=src, 
                                                    attention_mask=mask_src, 
                                                    do_sample=False, 
                                                    max_length=256, 
                                                    min_length=5,
                                                    num_beams=5,
                                                    no_repeat_ngram_size=5)
            return output_sequences
        else:
            outputs = self.model(src, attention_mask=mask_src, labels=labels)

        loss = outputs[0]; lm_logits = outputs[1]
        logging_info = self._log_generation_stats(lm_logits, tgt[:,1:], src)

        return loss, logging_info



class AbsSummarizer(nn.Module):
    def __init__(self, args, device, vocab_size, checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        #self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name)

        self.vocab_size = vocab_size
        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        print (self.vocab_size, len(self.original_tokenizer))
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)
        self.pad_id = self.original_tokenizer.pad_token_id

        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, 
                                        self.model.config.hidden_size, 
                                        device, 
                                        pre_trained_generator=self.model.lm_head)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            print ('Initialize parameters for generator...')
            '''
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            '''

        self.to(device)

    def _log_generation_stats(self, scores, target, src):
        pred = scores.max(2)[1]
        non_padding = target.ne(self.pad_id)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        log_info = {'num_non_padding':num_non_padding, 'num_correct':num_correct, 'n_docs':int(src.size(0))}
        return log_info

    def forward(self, src, tgt, mask_src, mask_tgt, run_decoder=True):

        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        if not run_decoder:
            #return {"encoder_outpus":top_vec, "encoder_attention_mask":mask_src}
            output_sequences = self.model.generate(input_ids=src, 
                                                    attention_mask=mask_src, 
                                                    do_sample=False, 
                                                    max_length=128, 
                                                    min_length=5,
                                                    num_beams=5,
                                                    no_repeat_ngram_size=5)
            return output_sequences

        # Decoding
        decoder_outputs = self.decoder(input_ids=tgt[:,:-1], 
                                       attention_mask=mask_tgt[:,:-1],
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src)

        #return decoder_outputs.last_hidden_state

        sequence_output = decoder_outputs.last_hidden_state
        sequence_output = sequence_output * (self.model.model_dim ** -0.5)
        lm_logits = self.model.lm_head(sequence_output)

        labels = tgt[:,1:]
        logging_info = self._log_generation_stats(lm_logits, labels, src)

        loss_fct = CrossEntropyLoss(ignore_index=self.pad_id)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.reshape(-1))

        return loss, logging_info



class SpectralReinforce(nn.Module):
    def __init__(self, args, device, pad_id, vocab_size, tokenizer, checkpoint=None):
        super(SpectralReinforce, self).__init__()
        self.args = args
        self.device = device
        #self.pad_id = pad_id

        self.abs_model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        print (self.vocab_size, len(self.original_tokenizer))
        if self.vocab_size > len(self.original_tokenizer):
            self.abs_model.resize_token_embeddings(self.vocab_size)
        self.pad_id = self.original_tokenizer.pad_token_id
        self.all_predicates = [i for i in range(len(self.original_tokenizer), self.vocab_size)]

        self.deterministic_graph = SpectralCluser(method = 'spectral_clustering',
                                                 assign_labels = 'discretize',
                                                 eigen_solver = 'arpack',
                                                 affinity = 'precomputed',
                                                 max_group_size = 10,
                                                 min_pair_freq = 15,
                                                 use_ratio = False,
                                                 filter_with_entities = True,
                                                 train_file = args.deterministic_graph_path)

        if self.args.nn_graph:
            self.predicate_graph = PairClassification(len(self.tokenizer),
                                                      pad_id,
                                                      self.args.nn_graph_d_model,
                                                      self.args.nn_graph_d_ff,
                                                      self.args.nn_graph_heads,
                                                      self.args.nn_graph_dropout,
                                                      num_inter_layers=self.args.nn_graph_nlayers)
        else:
            self.predicate_graph = nn.Parameter(torch.full((self.vocab_size, self.vocab_size), -7.0, device=self.device))

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.nll = nn.NLLLoss(ignore_index=self.pad_id, reduce=False)
        self.cls_loss = torch.nn.BCELoss(reduction='none')

        if self.args.reinforce_strong_baseline:
            self.baseline_predicate_graph = PairClassification(len(self.tokenizer),
                                                      pad_id,
                                                      self.args.nn_graph_d_model,
                                                      self.args.nn_graph_d_ff,
                                                      self.args.nn_graph_heads,
                                                      self.args.nn_graph_dropout,
                                                      num_inter_layers=self.args.nn_graph_nlayers)
            self.baseline_predicate_graph.load_state_dict(self.predicate_graph.state_dict())
            for param in self.baseline_predicate_graph.parameters():
                param.requires_grad = False

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)

        if args.train_predicate_graph_only:
            for param in self.abs_model.parameters():
                param.requires_grad = False
            if args.from_scratch:
                for p in self.predicate_graph.parameters():
                    nn.init.normal_(p.data, mean=-0.0, std=0.3)
        if args.pretrain_encoder_decoder:
            self.predicate_graph.requires_grad = False

        self.to(device)


    def _log_generation_stats(self, scores, target, src):
        pred = scores.max(1)[1]
        non_padding = target.ne(self.pad_id)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        log_info = {'num_non_padding':num_non_padding, 'num_correct':num_correct, 'n_docs':int(src.size(0))}
        return log_info


    def _log_classification_stats(self, scores, target):

        prediction = (scores > 0.5)
        num_correct = prediction.eq(target).sum().item()
        log_info = {'num_non_padding':scores.size(0), 'num_correct':num_correct, 'n_docs':scores.size(0)}

        return log_info


    def _pad(self, data, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [self.pad_id] * (width - len(d)) for d in data]
        return rtn_data


    def _examine_src_pairs(self, triple_i, triple_j):

        def _extract_entities(triple):
            sub_idx = triple.find('<SUB>')
            obj_idx = triple.find('<OBJ>')
            pred_idx = triple.find('<PRED>')
            sub = triple[sub_idx+5:pred_idx].strip()
            obj = triple[obj_idx+5:].strip()
            return sub, obj

        sub_i, obj_i = _extract_entities(triple_i)
        sub_j, obj_j = _extract_entities(triple_j)
        res = (len(set([sub_i, obj_i]) & set([sub_j, obj_j])) > 0)

        return res


    def _label_to_descrete_adjacency(self, lables):
        ntripls = len(lables)
        adjacency = torch.zeros((ntripls, ntripls), device=self.device)
        id_to_label = {}
        for i, label in enumerate(lables):
            id_to_label[i] = label
        for i in range(ntripls):
            for j in range(ntripls):
                if i == j:
                    label = id_to_label[i]
                    if lables.count(label) == 1:
                        adjacency[i][j] = 1
                else:
                    if id_to_label[i] == id_to_label[j]:
                        adjacency[i][j] = 1
        return adjacency


    def _get_adjacency_matrix(self, predicate_graph, predicates, 
                              pred_token, pred_token_mask):
        # get symmetric adjacency matrix
        linear_prob_matrix, linear_score_matrix = predicate_graph(pred_token, pred_token_mask)

        '''
        linear_ajacency_matrix = linear_prob_matrix
        adja = linear_ajacency_matrix.view(len(predicates), -1)

        if self.args.nn_graph_dropout > 0.0:
            symmatric_ajda = (torch.transpose(adja, 0, 1) + adja)/2
        else:
            symmatric_ajda = adja

        return symmatric_ajda
        '''

        linear_ajacency_matrix = linear_score_matrix
        linear_ajacency_matrix = linear_ajacency_matrix.view(len(predicates), -1)
        adja = self.softmax(linear_ajacency_matrix)

        return adja


    def run_spectral(self, predicates, symmatric_ajda, n_clusters, 
                     src_str=None, pred_str=None, 
                     run_bernoulli=True, 
                     adja_threshold=-1):

        if len(predicates) == 1:
            return np.array([0]), symmatric_ajda

        if run_bernoulli:
            '''
            # sample edge and make it symmetric and fully connected
            sampled_edges = torch.bernoulli(symmatric_ajda)
            triu_index_i, triu_index_j = torch.triu_indices(len(predicates), len(predicates))
            sampled_edges[triu_index_j, triu_index_i] = sampled_edges[triu_index_i, triu_index_j]
            for i in range(len(predicates)):
                if sum(sampled_edges[i]) == 0:
                    sampled_edges[i][i] = 1
            #print (symmatric_ajda)
            #print (sampled_edges)
            adjacency_matrix = symmatric_ajda * sampled_edges
            '''
            m = RelaxedBernoulli(torch.tensor([self.args.reinforce_bernoulli_temp], device=self.device), symmatric_ajda)
            adjacency_matrix = m.rsample()
            adjacency_matrix = (torch.transpose(adjacency_matrix, 0, 1) + adjacency_matrix)/2
        else:
            adjacency_matrix = symmatric_ajda
            adjacency_matrix[adjacency_matrix < adja_threshold] = 0.0

        adjacency_matrix_numpy = adjacency_matrix.cpu().detach().numpy()

        # during testing, using the entity link bias to constrain the clustering 
        for i, head_1 in enumerate(predicates):
            for j, head_2 in enumerate(predicates):
                if self.args.test_entity_link and \
                        src_str is not None and \
                        (not self._examine_src_pairs(src_str[i], src_str[j])):
                    adjacency_matrix_numpy[i][j] = 0

        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels='discretize',
                                        eigen_solver='arpack',
                                        affinity='precomputed').fit(adjacency_matrix_numpy)
        #print (adjacency_matrix)
        #print (clustering.labels_)
        return clustering.labels_, adjacency_matrix


    def run_random(self, predicates, n_clusters):
        labels = [random.sample(range(n_clusters), 1)[0] for item in predicates]
        return labels


    def run_discriministic(self, src, preds, n_clusters):
        labels = self.deterministic_graph.get_aggragation_lable(preds, src, n_clusters)
        return labels


    def calculate_graph_prob_v0(self, pred_groups, predicates, adjacency_matrix):

        linear_ajacency_matrix = adjacency_matrix.reshape(-1)

        sub_graph = {}; idx = 0
        for i, head_i in enumerate(predicates):
            sub_graph[head_i] = {}
            for j, head_j in enumerate(predicates):
                sub_graph[head_i][head_j] = linear_ajacency_matrix[idx]
                idx += 1

        probs = []
        for group in pred_groups:
            in_log_prob = []
            for i, head_1 in enumerate(group):
                for j, head_2 in enumerate(group):
                    if head_1 != head_2:
                        likelihood = sub_graph[head_1][head_2]
                        #print (head_1, head_2, likelihood)
                        log_likelihood = torch.log(likelihood)
                        in_log_prob.append(log_likelihood)

                    elif len(group) == 1:
                        likelihood = sub_graph[head_1][head_2]
                        if self.args.test_no_single_pred_score:
                            likelihood = torch.tensor(1.0, device=self.device)
                        log_likelihood = torch.log(likelihood)
                        in_log_prob.append(log_likelihood)

            if len(in_log_prob) > 0:
                if self.args.calculate_graph_prob_method == 'min':
                    in_log_prob = min(in_log_prob)
                else:
                    in_log_prob = sum(in_log_prob)/len(in_log_prob)
                    
                probs.append(in_log_prob)

        return probs


    def calculate_graph_prob(self, pred_groups, predicates, adjacency_matrix):

        #print (adjacency_matrix)
        #print (predicates)
        #print (pred_groups)
        '''
        for i in range(adjacency_matrix.size(0)):
            for j in range(adjacency_matrix.size(1)):
                if i == j:
                    continue
                print (float(adjacency_matrix[i][j]))
        '''

        P_ij = {}; idx = 0
        linear_ajacency_matrix = adjacency_matrix.reshape(-1)
        for i, head_i in enumerate(predicates):
            P_ij[head_i] = {}
            for j, head_j in enumerate(predicates):
                P_ij[head_i][head_j] = linear_ajacency_matrix[idx]
                idx += 1

        probs = []
        for group in pred_groups:
            #print ('predicates', predicates)
            #print ('pred_groups', pred_groups)
            #print ('group', group)
            in_prob = []
            for i, head_1 in enumerate(group):
                for j, head_2 in enumerate(group):
                    if head_1 != head_2:
                        #print ('IN:', head_1, head_2)
                        likelihood = P_ij[head_1][head_2]
                        in_prob.append(likelihood)

                    elif len(group) == 1:
                        likelihood = P_ij[head_1][head_2]
                        #print ('IN:', head_1, head_2)
                        if self.args.test_no_single_pred_score:
                            likelihood = torch.tensor(1.0, device=self.device)
                        in_prob.append(likelihood)

            out_prob = []
            for head_1 in group:
                for head_2 in predicates:
                    if len(group) > 1 and head_1 == head_2:
                        #print ('OUT:', head_1, head_2)
                        likelihood = P_ij[head_1][head_2]
                        out_prob.append(likelihood)
                    elif head_2 not in group:
                        #print ('OUT:', head_1, head_2)
                        likelihood = P_ij[head_1][head_2]
                        out_prob.append(likelihood)

            if len(in_prob) > 0:
                if self.args.calculate_graph_prob_method == 'min':
                    in_prob = min(in_prob)
                    log_prob = torch.log(in_prob)
                else:
                    #in_prob = sum(in_prob)
                    #out_prob = sum(out_prob)
                    #log_prob = torch.log(in_prob/out_prob)

                    log_prob = sum([torch.log(in_p) for in_p in in_prob]) + sum([torch.log(1-out_p) for out_p in out_prob])

                    #log_in_prob = sum([torch.log(item) for item in in_prob])
                    #log_out_prob = sum([torch.log(item) for item in out_prob])
                    #log_prob = log_in_prob - log_out_prob
                    
                probs.append(log_prob)

       # print ('\n\n')

        return probs


    def calculate_graph_prob_v2(self, labels, adjacency_matrix):
        #mean, var = torch.var_mean(adjacency_matrix)
        #if adjacency_matrix.size(0) > 1:
        #    print (adjacency_matrix)
        #    print ('Sta:', float(torch.min(adjacency_matrix)), float(torch.max(adjacency_matrix)), float(mean), float(var))
        #    print ('\n')

        selection_matrix = self._label_to_descrete_adjacency(labels)
        LL_graph = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                A_ij = selection_matrix[i][j]
                P_ij = adjacency_matrix[i][j]
                LL = A_ij * torch.log(P_ij) + (1-A_ij) * torch.log(1-P_ij)
                LL_graph.append(LL)
        sum_LL = sum(LL_graph)

        if self.args.calculate_graph_prob_method == 'min':
            if self.args.test_no_single_pred_score:
                adjacency_matrix.fill_diagonal_(1)
            selected_edges_index = selection_matrix.nonzero(as_tuple=True)
            selected_edges = adjacency_matrix[selected_edges_index]
            group_num = len(set(labels))
            #print (adjacency_matrix)
            #print (labels)
            #print (selection_matrix)
            #print (min(selected_edges))
            #print ('\n')
            return [min(selected_edges) for i in range(group_num)]

        return [sum_LL]


    def run_clustering(self, src, preds, n_clusters, p2s, 
                       pred_token, pred_token_mask, 
                       mode='spectral', 
                       src_str=None, pred_str=None, 
                       run_bernoulli=True,
                       adja_threshold=-1):
        
        if mode == 'spectral_baseline':
            adjacency_matrix = self._get_adjacency_matrix(self.baseline_predicate_graph, preds, pred_token, pred_token_mask)
            if n_clusters == 1:
                mode = 'random'
        elif mode == 'threshold_baseline':
            adjacency_matrix = self._get_adjacency_matrix(self.predicate_graph, preds, pred_token, pred_token_mask)
            if n_clusters == 1:
                mode = 'random'
        else:
            adjacency_matrix = self._get_adjacency_matrix(self.predicate_graph, preds, pred_token, pred_token_mask)
            #print ('[UPDATE]', adjacency_matrix, adjacency_matrix.requires_grad)

        ##########################################################
        # run clustering
        ##########################################################

        sample_matrix = None
        if mode == 'gold':
            pred_to_group = {}
            for i, group in enumerate(p2s):
                for pred in group:
                    pred_to_group[pred] = i
            labels = []
            for pred in preds:
                if pred not in pred_to_group:
                    continue
                labels.append(pred_to_group[pred])

        elif mode == 'full_src':
            labels = [0 for s in src]
           
        elif mode == 'random':
            labels = self.run_random(preds, n_clusters)
            while len(set(labels)) < n_clusters:
                labels = self.run_random(preds, n_clusters)
            if n_clusters == 1:
                labels = [random.sample([0, -1], 1)[0] for i in range(len(labels))]
                while sum(labels) * (-1) == len(labels) or sum(labels) == 0:
                    labels = [random.sample([0, -1], 1)[0] for i in range(len(labels))]

        elif mode == 'random_test':
            labels = self.run_random(preds, n_clusters)
            while len(set(labels)) < n_clusters:
                labels = self.run_random(preds, n_clusters)

        elif mode == 'discriministic':
            labels = self.run_discriministic(src_str, pred_str, n_clusters)
            #labels = labels.tolist()

        else:
            if mode == 'spectral_baseline':
                run_bernoulli = False
            elif mode == 'threshold_baseline':
                run_bernoulli = False
                adja_threshold = self.args.reinforce_baseline_adja_threshold
            labels, sample_matrix = self.run_spectral(preds, adjacency_matrix, n_clusters, 
                                                      src_str, pred_str, 
                                                      run_bernoulli=run_bernoulli, 
                                                      adja_threshold=adja_threshold)
            labels = labels.tolist()

        ##########################################################
        # group predicates and src based on the cluster method
        ##########################################################
        pred_groups = [[] for i in range(n_clusters)]
        tmp_src_groups = [[] for i in range(n_clusters)]
        for i, label in enumerate(labels):
            if label == -1:
                continue
            pred_groups[label].append(preds[i])
            tmp_src_groups[label].append(src[i])

        src_groups = [[] for i in range(n_clusters)]
        for i in range(n_clusters):
            sg = tmp_src_groups[i]
            if self.args.shuffle_src:
                random.shuffle(sg)
            for item in sg:
                src_groups[i].extend(item)

        pred_str_groups = None; src_str_groups = None
        if (src_str is not None) and (pred_str is not None):
            pred_str_groups = [[] for i in range(n_clusters)]
            src_str_groups = [[] for i in range(n_clusters)]
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                pred_str_groups[label].append(pred_str[i])
                src_str_groups[label].append(src_str[i])


        ##########################################################
        # calculate graph probability
        ##########################################################
        if mode == 'discriministic':
            graph_prob = self.deterministic_graph.calculate_graph_score(labels, pred_str, n_clusters)
        elif mode == 'full_src':
            graph_prob = [0.0]
        elif mode == 'random_test':
            graph_prob = [0.0 for i in range(n_clusters)]
        else:
            if sample_matrix is not None:
                graph_prob = self.calculate_graph_prob(pred_groups, preds, sample_matrix)
            else:
                graph_prob = self.calculate_graph_prob(pred_groups, preds, adjacency_matrix)
            #§graph_prob = self.calculate_graph_prob(pred_groups, preds, adjacency_matrix)
            #graph_prob = self.calculate_graph_prob(labels, adjacency_matrix)

        '''
        print (n_clusters)
        print ( mode)
        print (src)
        print (labels)
        print (src_groups)
        print (pred_groups)
        print (graph_prob)
        print (src_str_groups)
        print (pred_str_groups)
        print ('')
        '''
        return src_groups, pred_groups, graph_prob, src_str_groups, pred_str_groups, n_clusters


    def hungrian_alignment(self, pred_groups, tgt, mask_tgt, ctgt, mask_ctgt, mask_ctgt_loss, p2s, n_clusters):
        # Hungrian alignment
        #print (pred_groups, p2s)
        predicate_to_slot = pred_groups
        predicate_to_sentence = p2s
        slot_to_sentence_costs = np.zeros((n_clusters, n_clusters)) # slot to sentence alignments
        for j in range(len(predicate_to_slot)):
            for k in range(len(predicate_to_sentence)):
                overlap = len(set(predicate_to_slot[j]) & set(predicate_to_sentence[k]))*2+1
                slot_to_sentence_costs[j][k] = overlap

        slot_to_sentence_costs = slot_to_sentence_costs.max() - slot_to_sentence_costs
        row_ind, col_ind = linear_sum_assignment(slot_to_sentence_costs)

        new_tgt = torch.stack([tgt[idx] for idx in col_ind])
        new_tgt_mask = torch.stack([mask_tgt[idx] for idx in col_ind])

        new_ctgt = torch.stack([ctgt[idx] for idx in col_ind])
        new_ctgt_mask = torch.stack([mask_ctgt[idx] for idx in col_ind])
        new_ctgt_mask_loss = torch.stack([mask_ctgt_loss[idx] for idx in col_ind])

        return new_tgt, new_tgt_mask, new_ctgt, new_ctgt_mask, new_ctgt_mask_loss


    def forward_generator(self, src, tgt, mask_tgt, ctgt, 
                          mask_ctgt, mask_ctgt_loss, 
                          preds, pred_tokens, pred_mask_tokens, 
                          p2s, nsent, run_decoder=True, mode='spectral'):

        raw_tgt = tgt

        parallel_src = []
        parallel_tgt = []
        parallel_tgt_mask = []
        parallel_tgt_mask_loss = []
        parallel_ctgt = []
        parallel_ctgt_mask = []
        parallel_graph_probs = []
        ngroups = []

        for i in range(len(preds)):
            s = src[i] # src sentences
            t = tgt[i] # tgt sentences
            m_t = mask_tgt[i] # mask for tgt sentences
            ct = ctgt[i] # tgt sentences with previous tokens as conditions
            m_ct = mask_ctgt[i] # mask for tgt sentences with previous tokens as conditions
            m_ct_l = mask_ctgt_loss[i] # mask for tgt sentences with previous tokens as conditions for loss calculation only

            p = preds[i]
            p_s = p2s[i]
            p_tok = pred_tokens[i]
            p_tok_m = pred_mask_tokens[i]

            n_clusters = t.size(0)
            if mode == 'full_src':
                n_clusters = 1

            # run clustering
            src_groups, pred_groups, graph_probs, _, _, n_clusters = self.run_clustering(s, p, n_clusters, p_s, p_tok, p_tok_m, mode=mode)

            # run cluster-to-output_sentence alignment
            if mode == 'gold':
                new_tgt, new_tgt_mask, new_ctgt, new_ctgt_mask, new_mask_ctgt_loss = t, m_t, ct, m_ct, m_ct_l
            else:
                res = self.hungrian_alignment(pred_groups, t, m_t, ct, m_ct, m_ct_l, p_s, n_clusters)
                new_tgt, new_tgt_mask, new_ctgt, new_ctgt_mask, new_mask_ctgt_loss = res

            '''
            if n_clusters > 1:
                print (p_s)
                print ('src', s)
                print ('tgt', t)
                print ('pred', p)
                print ('src_groups', src_groups)
                print ('new_tgt', new_tgt)
                print ('new_tgt_mask', new_tgt_mask)
                print ('new_ctgt', new_ctgt)
                print ('new_ctgt_mask', new_ctgt_mask)
                print ('new_mask_ctgt_loss', new_mask_ctgt_loss)
                print ('\n\n')
            '''

            parallel_src.extend(src_groups)
            parallel_tgt.append(new_tgt)
            parallel_tgt_mask.append(new_tgt_mask)
            parallel_ctgt.append(new_ctgt)
            parallel_ctgt_mask.append(new_ctgt_mask)
            parallel_tgt_mask_loss.append(new_mask_ctgt_loss)
            parallel_graph_probs.extend(graph_probs)
            ngroups.append(t.size(0))

        # Preparing encoder
        src = torch.tensor(self._pad(parallel_src), device=self.device)
        mask_src = ~(src == self.pad_id)

        # Preparing decoder
        tgt = torch.cat(parallel_tgt)
        mask_tgt = torch.cat(parallel_tgt_mask)
        ctgt = torch.cat(parallel_ctgt)
        mask_ctgt = torch.cat(parallel_ctgt_mask)
        mask_ctgt_loss = torch.cat(parallel_tgt_mask_loss)
        gtruth = tgt
        labels = tgt

        if self.args.conditional_decoder:
            tgt = ctgt
            mask_tgt = mask_ctgt
            labels = tgt * mask_ctgt_loss + self.pad_id * (~ mask_ctgt_loss)
            gtruth = tgt * mask_ctgt_loss + self.pad_id * (~ mask_ctgt_loss)

        gtruth = gtruth[:, 1:].contiguous()
        labels = labels[:, 1:].contiguous()
        tgt = tgt[:, :-1].contiguous()
        mask_tgt = mask_tgt[:, :-1].contiguous()

        labels[labels == self.pad_id] = -100
        outputs = self.abs_model(input_ids=src, attention_mask=mask_src, 
                                 decoder_input_ids=tgt,
                                 decoder_attention_mask=mask_tgt,
                                 labels=labels)
        scores = outputs.logits # [batch_size, tgt_length, vocab_size]

        scores = scores.reshape(-1, scores.size(2))
        gtruth = gtruth.reshape(-1)
        logging_info = self._log_generation_stats(scores, gtruth, src) # log_info

        loss = outputs.loss # single score

        if self.args.pretrain_encoder_decoder:
            return loss, None, logging_info

        loss_fct = CrossEntropyLoss(ignore_index=self.pad_id, reduce=False)
        log_neglikelihood = loss_fct(scores, gtruth)
        log_likelihood = (-1) * log_neglikelihood

        # log
        '''
        print ('mode:', mode)
        for i in range(output.size(0)):
            gtruth = gtruth.view(src.size(0), -1)
            #print (' '.join(self.tokenizer.convert_ids_to_tokens(src[i])), '|||||||',  ' '.join(self.tokenizer.convert_ids_to_tokens(gtruth[i])))
            print (' '.join(self.tokenizer.convert_ids_to_tokens(src[i])), '|||||||',  self.tokenizer.decode(gtruth[i], skip_special_tokens=True))
        '''

        log_likelihood = log_likelihood.view(tgt.size(0),-1)
        log_likelihood = torch.mean(log_likelihood, dim=1)

        weights = torch.stack(parallel_graph_probs)

        # post process the generation likelihood
        cur_id = 0
        processed_log_likelihood = []
        for nline in nsent:
            processed_log_likelihood.append(log_likelihood[cur_id:(cur_id+nline)].min())
            cur_id += nline
        log_likelihood = torch.stack(processed_log_likelihood)

        if weights.size() == sum(nsent):
            cur_id = 0
            processed_weights = []
            for nline in nsent:
                processed_weights.append(weights[cur_id:(cur_id+nline)].sum())
                cur_id += nline
            weights = torch.stack(processed_weights)

        # log
        #print (log_likelihood, weights)
        #print ('\n')

        return log_likelihood, weights, logging_info


    def forward_cls(self, pred_tokens, pred_mask_tokens, aggregation_labels, nsents):

        tokens = torch.cat(pred_tokens)
        masks = torch.cat(pred_mask_tokens)
        labels = torch.cat([item.unsqueeze(1) for item in aggregation_labels]).squeeze(-1)

        sent_scores, _ = self.predicate_graph(tokens, masks)
        loss = self.cls_loss(sent_scores, labels.float())

        logging_info = self._log_classification_stats(sent_scores, labels) # log_info

        return loss, logging_info


    def forward(self, src, tgt, mask_tgt,
                ctgt, mask_ctgt, mask_ctgt_loss, 
                preds, pred_tokens, pred_mask_tokens, 
                p2s, nsent, aggregation_labels=None,
                run_decoder=True, mode='spectral'):

        if mode == 'nn':
            ret = self.forward_cls(pred_tokens, pred_mask_tokens, aggregation_labels, nsent)
        else:
            ret = self.forward_generator(src, tgt, mask_tgt, ctgt,
                                         mask_ctgt, mask_ctgt_loss,
                                         preds, pred_tokens, pred_mask_tokens,
                                         p2s, nsent, run_decoder=run_decoder,
                                         mode=mode)
        return ret
