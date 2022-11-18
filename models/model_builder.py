import copy, random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from models.optimizers import Optimizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import LongformerModel, BertModel
from models.encoder import SentenceClassification, ClusterClassification, TransformerEncoder, Classifier
from models.slot_attn import SlotAttention, SoftKMeans
from sentence_transformers.models import Pooling

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    if checkpoint is not None:
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

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

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

    else:
        optim = Optimizer(
            args.optim, args.lr_encdec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_encdec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('planner')]
    optim.set_parameters(params)

    return optim


def build_optim_planner(args, model, checkpoint):
    """ Build optimizer """

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
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_planner, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_planner)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('planner')]
    optim.set_parameters(params)

    return optim


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, vocab_size, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)

        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        self.planning_layer = SentenceClassification(self.model.config.hidden_size, 
                                                     args.ext_ff_size, 
                                                     args.ext_heads, 
                                                     args.ext_dropout, 
                                                     args.ext_layers)

        if checkpoint is not None:
            print ('Load parameters from ext_finetune...')
            #self.load_state_dict(checkpoint['model'], strict=True)
            tree_params = [(n[15:], p) for n, p in checkpoint['model'].items() if n.startswith('planning_layer')]
            self.planning_layer.load_state_dict(dict(tree_params), strict=True)
            tree_params = [(n[8:], p) for n, p in checkpoint['model'].items() if n.startswith('encoder')]
            self.encoder.load_state_dict(dict(tree_params), strict=True)
        else:
            if self.planning_layer is not None:
                if args.param_init != 0.0:
                    for p in self.planning_layer.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                if args.param_init_glorot:
                    for p in self.planning_layer.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)

        if args.freeze_encoder_decoder:
            for param in self.model.parameters():
                param.requires_grad = False

        self.to(device)


    def forward(self, src, tgt, mask_src, mask_tgt, clss, mask_cls):
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        # return transformers.modeling_outputs.BaseModelOutput
        top_vec = encoder_outputs.last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]

        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores, aj_matrixes = self.planning_layer(sents_vec, mask_cls)

        return sent_scores, mask_cls, aj_matrixes, top_vec



def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)
    return generator



class AbsSummarizer(nn.Module):
    def __init__(self, args, device, cls_token_id, vocab_size, checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = vocab_size
        self.cls_token_id = cls_token_id

        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        print (self.vocab_size, len(self.original_tokenizer))
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, self.model.config.hidden_size, device)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            print ('Initialize parameters for generator...')
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

        self.to(device)


    def forward(self, src, tgt, mask_src, mask_tgt, run_decoder=True):

        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        if not run_decoder:
            return {"encoder_outpus":top_vec, "encoder_attention_mask":mask_src}

        # Decoding
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src)

        return decoder_outputs.last_hidden_state


class ParagraphMultiClassifier(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ParagraphMultiClassifier, self).__init__()
        self.args = args
        self.device = device
        #self.local_bert = BertModel.from_pretrained(args.model_name)
        self.local_bert = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=True, return_dict=True)
        self.sentence_pooling = Pooling(self.local_bert.config.hidden_size)
        self.cluster_pooling = Pooling(self.local_bert.config.hidden_size)
        self.classifiers = ClusterClassification(self.local_bert.config.hidden_size,
                                                 self.args.ext_ff_size,
                                                 self.args.ext_heads,
                                                 self.args.ext_dropout,
                                                 num_inter_layers=self.args.ext_layers)

        self.model_hidden_size = self.local_bert.config.hidden_size

        if checkpoint is not None:
            print ('Load parameters from ext_finetune...')
            params = [(n[12:], p) for n, p in checkpoint['model'].items() if n.startswith('classifiers')]
            self.classifiers.load_state_dict(dict(params), strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.classifiers.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.classifiers.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        # To save memory
        for param in self.local_bert.parameters():
            param.requires_grad = False

        self.to(device)

    def pool_cluster_embeddings(self, sentence_embeddings, cluster_sizes):
        cluster_numbers = sum([len(ex) for ex in cluster_sizes])
        maximum_cluster_size = max([max(ex) for ex in cluster_sizes])
        embedding_size = self.model_hidden_size
        cluster_embeddings = torch.zeros((cluster_numbers, maximum_cluster_size, embedding_size), device=self.device)
        cluster_masks = torch.full((cluster_numbers, maximum_cluster_size), False, device=self.device)
        cluster_idx = 0; start_idx = 0
        for example_cluster_sizes in cluster_sizes:
            for cluster_size in example_cluster_sizes:
                sid = start_idx
                eid = start_idx + cluster_size
                cluster_embeddings[cluster_idx][:cluster_size] = sentence_embeddings[sid:eid]
                cluster_masks[cluster_idx][:cluster_size] = True
                start_idx += cluster_size
                cluster_idx += 1
        res = self.cluster_pooling({'token_embeddings':cluster_embeddings, 'attention_mask':cluster_masks})
        return res['sentence_embedding']
        

    def cluster_classification(self, cluster_embeddings, cluster_sizes):
        example_num = len(cluster_sizes)
        max_cluster_num = max([len(example) for example in cluster_sizes])
        embedding_size = self.model_hidden_size
        example_embeddings = torch.zeros((example_num, max_cluster_num, embedding_size), device=self.device)
        example_masks = torch.full((example_num, max_cluster_num), False, device=self.device)
        example_idx = 0; start_idx = 0
        for example_cluster_size in cluster_sizes:
            sid = start_idx
            eid = start_idx + len(example_cluster_size)
            example_embeddings[example_idx][:len(example_cluster_size)] = cluster_embeddings[sid:eid]
            example_masks[example_idx][:len(example_cluster_size)] = True
            start_idx += len(example_cluster_size)
            example_idx += 1
        verd_scores, pros_scores, cons_scores = self.classifiers(example_embeddings, example_masks)
        return verd_scores, pros_scores, cons_scores, example_masks


    def forward(self, src, mask_src, cluster_sizes):
        res = self.local_bert(input_ids=src, attention_mask=mask_src)
        token_embeddings = res.hidden_states[-1]
        sentence_embeddings = token_embeddings[:, 0, :]
        '''
        token_embeddings = self.local_bert(input_ids=src, attention_mask=mask_src)
        res = self.sentence_pooling({'token_embeddings':token_embeddings.last_hidden_state, 'attention_mask':mask_src})
        # number of sentences * embedding size 
        sentence_embeddings = res['sentence_embedding']
        '''
        # number of clusters * embedding size 
        cluster_embeddings = self.pool_cluster_embeddings(sentence_embeddings, cluster_sizes)
        verd_scores, pros_scores, cons_scores, example_masks = self.cluster_classification(cluster_embeddings, cluster_sizes)

        return verd_scores, pros_scores, cons_scores, example_masks



class ClusterMultiClassifier(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ClusterMultiClassifier, self).__init__()
        self.args = args
        self.device = device

        self.sentence_encoder = AutoModelForSequenceClassification.from_pretrained(args.model_name, 
                                                                             output_hidden_states=True, 
                                                                             return_dict=True)

        #self.cls_tokens_emb = torch.rand((3, self.sentence_encoder.config.hidden_size), device=self.device)
        self.cls_tokens_emb = nn.Embedding(3, self.sentence_encoder.config.hidden_size, device=self.device, max_norm=True)

        self.cluster_encoder = TransformerEncoder(self.sentence_encoder.config.hidden_size,
                                                 self.args.ext_ff_size,
                                                 self.args.ext_heads,
                                                 self.args.ext_dropout,
                                                 num_inter_layers=self.args.ext_layers)

        self.classifier_verd = Classifier(self.sentence_encoder.config.hidden_size)
        self.classifier_pros = Classifier(self.sentence_encoder.config.hidden_size)
        self.classifier_cons = Classifier(self.sentence_encoder.config.hidden_size)


        if checkpoint is not None:
            print ('Load parameters from ext_finetune...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.cluster_encoder.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in self.classifier_verd.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in self.classifier_pros.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in self.classifier_cons.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.cluster_encoder.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in self.classifier_verd.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in self.classifier_pros.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in self.classifier_cons.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        # To save memory
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

        self.to(device)


    def prepare_for_cluster_embeddings(self, sentence_embeddings, cluster_sizes):
        n_example = sum([len(ex) for ex in cluster_sizes])
        maximum_cluster_size = max([max(ex) for ex in cluster_sizes])+3
        embedding_size = self.sentence_encoder.config.hidden_size
        cluster_embeddings = torch.zeros((n_example, maximum_cluster_size, embedding_size), device=self.device)
        cluster_masks = torch.full((n_example, maximum_cluster_size), False, device=self.device)
        cluster_idx = 0; start_idx = 0
        for item in cluster_sizes:
            cluster_size = item[0]
            sid = start_idx
            eid = start_idx + cluster_size
            cluster_embeddings[cluster_idx][:3] = self.cls_tokens_emb.weight
            cluster_embeddings[cluster_idx][3:cluster_size+3] = sentence_embeddings[sid:eid]
            cluster_masks[cluster_idx][:cluster_size+3] = True
            start_idx += cluster_size
            cluster_idx += 1
        return cluster_embeddings, cluster_masks
        

    def forward(self, src, mask_src, cluster_sizes):
        # sentence encoding
        res = self.sentence_encoder(input_ids=src, attention_mask=mask_src)
        token_embeddings = res.hidden_states[-1]
        sentence_embeddings = token_embeddings[:, 0, :] # number of clusters * embedding size 

        # cluster encoding
        cluster_embeddings, cluster_masks = self.prepare_for_cluster_embeddings(sentence_embeddings, cluster_sizes)
        cluster_embeddings, cluster_masks = self.cluster_encoder(cluster_embeddings, cluster_masks)

        # classification
        verd_scores = self.classifier_verd(cluster_embeddings[:,0,:], cluster_masks[:,0])
        pros_scores = self.classifier_pros(cluster_embeddings[:,1,:], cluster_masks[:,1])
        cons_scores = self.classifier_cons(cluster_embeddings[:,2,:], cluster_masks[:,2])
        verd_scores = verd_scores.unsqueeze(1)
        pros_scores = pros_scores.unsqueeze(1)
        cons_scores = cons_scores.unsqueeze(1)

        example_masks = cluster_masks[:,0].unsqueeze(1)

        return verd_scores, pros_scores, cons_scores, example_masks



class ClusterLongformerClassifier(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint):
        super(ClusterLongformerClassifier, self).__init__()
        self.args = args
        self.device = device

        self.cluster_encoder = LongformerModel.from_pretrained(args.model_name)

        self.classifier_verd = Classifier(self.cluster_encoder.config.hidden_size)
        self.classifier_pros = Classifier(self.cluster_encoder.config.hidden_size)
        self.classifier_cons = Classifier(self.cluster_encoder.config.hidden_size)

        self.cls_token_ids = tokenizer.convert_tokens_to_ids([args.cls_verdict, args.cls_pros, args.cls_cons])
        self.cls_token_ids = torch.tensor(self.cls_token_ids, device=self.device)
        self.cls_token_mask = torch.tensor([True, True, True], device=self.device)

        if checkpoint is not None:
            print ('Load parameters from ext_finetune...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.classifier_verd.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in self.classifier_pros.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in self.classifier_cons.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.classifier_verd.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in self.classifier_pros.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in self.classifier_cons.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)


    def frame_input(self, src, src_mask):
        batch_size = src.size(0)
        cls_token_ids = self.cls_token_ids.repeat(batch_size, 1)
        new_src = torch.cat((cls_token_ids, src), dim=1)
        cls_token_mask = self.cls_token_mask.repeat(batch_size, 1)
        new_mask = torch.cat((cls_token_mask, src_mask), dim=1)
        global_mask = torch.zeros(new_src.size(), device=self.device)
        global_mask[:, :3] = 1
        return new_src, new_mask, global_mask
        

    def forward(self, src, mask_src, cluster_sizes):
        src, mask_src, global_mask = self.frame_input(src, mask_src)
        res = self.cluster_encoder(input_ids=src, attention_mask=mask_src, global_attention_mask=global_mask)
        cluster_embeddings = res.last_hidden_state

        verd_scores = self.classifier_verd(cluster_embeddings[:,0,:], mask_src[:,0])
        pros_scores = self.classifier_pros(cluster_embeddings[:,1,:], mask_src[:,1])
        cons_scores = self.classifier_cons(cluster_embeddings[:,2,:], mask_src[:,2])
        verd_scores = verd_scores.unsqueeze(1)
        pros_scores = pros_scores.unsqueeze(1)
        cons_scores = cons_scores.unsqueeze(1)

        example_masks = mask_src[:,0].unsqueeze(1)

        return verd_scores, pros_scores, cons_scores, example_masks


class ClusterBinarySelection(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint):
        super(ClusterBinarySelection, self).__init__()
        self.args = args
        self.device = device
        self.cluster_encoder = LongformerModel.from_pretrained(args.model_name)
        self.classifier = Classifier(self.cluster_encoder.config.hidden_size)

        if checkpoint is not None:
            print ('Load parameters from ext_finetune...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.classifier.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.classifier.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, mask_src, cluster_sizes):
        global_mask = torch.zeros(src.size(), device=self.device)
        global_mask[:, 0] = 1
        res = self.cluster_encoder(input_ids=src, attention_mask=mask_src, global_attention_mask=global_mask)
        cluster_embeddings = res.last_hidden_state

        scores = self.classifier(cluster_embeddings[:,0,:], mask_src[:,0])
        scores = scores.unsqueeze(1)

        example_masks = mask_src[:,0].unsqueeze(1)

        if self.args.cls_type == 'verdict' or self.args.cls_type == 'select':
            return scores, None, None, example_masks
        if self.args.cls_type == 'pros':
            return None, scores, None, example_masks
        if self.args.cls_type == 'cons':
            return None, None, scores, example_masks


class SentimentClassifier(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint):
        super(SentimentClassifier, self).__init__()
        self.args = args
        self.device = device
        self.cluster_encoder = LongformerModel.from_pretrained(args.model_name)
        self.classifier = Classifier(self.cluster_encoder.config.hidden_size, output_size=3)

        if checkpoint is not None:
            print ('Load parameters from ext_finetune...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.classifier.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.classifier.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, mask_src, cluster_sizes):
        global_mask = torch.zeros(src.size(), device=self.device)
        global_mask[:, 0] = 1
        res = self.cluster_encoder(input_ids=src, attention_mask=mask_src, global_attention_mask=global_mask)
        cluster_embeddings = res.last_hidden_state
        scores = self.classifier(cluster_embeddings[:,0,:])

        verdict_scores = scores[:, 0].unsqueeze(1)
        pros_scores = scores[:, 1].unsqueeze(1)
        cons_scores = scores[:, 2].unsqueeze(1)

        example_masks = mask_src[:,0].unsqueeze(1)

        return verdict_scores, pros_scores, cons_scores, example_masks


class SlotAttnAggragator(nn.Module):
    def __init__(self, args, device, vocab_size, checkpoint=None, pretrained_checkpoint=None):
        super(SlotAttnAggragator, self).__init__()
        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.original_vocab_size = len(self.original_tokenizer)

        if pretrained_checkpoint is not None:
            self.model = pretrained_checkpoint
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
            self.generator = self.model.generator
            #if self.args.slot_sample_mode=='marginal':
            #    self.decoder.embed_tokens.weight[self.original_vocab_size:, :].data = self.encoder.embed_tokens.weight[self.original_vocab_size:, :].data
            model_dim = 768
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
            if self.vocab_size > self.original_vocab_size:
                self.model.resize_token_embeddings(self.vocab_size)
            self.encoder = self.model.get_encoder()
            self.decoder = self.model.get_decoder()
            model_dim = self.model.config.hidden_size
            self.generator = get_generator(self.vocab_size, model_dim, device)

        if self.args.cluster_algorithm == 'soft_kmeans':
            self.planner = SoftKMeans(args.slot_num_slots, model_dim, args.slot_iters, args.slot_eps, model_dim)
        else:
            self.planner = SlotAttention(args.slot_num_slots, model_dim, args.slot_iters, args.slot_eps, model_dim)
        self.planner_emb = nn.Embedding(self.decoder.embed_tokens.num_embeddings, self.decoder.embed_tokens.embedding_dim)
        self.planner_emb.weight[self.original_vocab_size:, :].data = self.encoder.embed_tokens.weight[self.original_vocab_size:, :].data

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            print ('Initialize parameters for generator...')
            if pretrained_checkpoint is None:
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()
            for p in self.planner.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

        if args.freeze_encoder_decoder:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.generator.parameters():
                param.requires_grad = False

        self.to(device)


    def forward(self, src, tgt, pred, p2s, mask_src, mask_tgt, nsent, train_progress=1):

        # Run encoder
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        # Run slot attn
        slot_embs = []; new_tgts = []; new_tgt_masks = []
        entropy_map = {}
        for i in range(len(nsent)):
            # for the i-th example, run slot attention
            #p_emb = self.decoder.embed_tokens(pred[i])
            p_emb = self.planner_emb(pred[i])
            p_emb = p_emb.unsqueeze(dim=0) # (batch_size=1, pred_num, dim)
            s_embs, s_attn, raw_attn_scores = self.planner(p_emb, num_slots=nsent[i]) # (batch_size=1, slot_num, dim), (batch_size=1, slot_num, pred_num)
            s_embs = s_embs.squeeze(dim=0) # (slot_num, dim)
            s_attn = s_attn.squeeze(dim=0) # (slot_num, dim)
            raw_attn_scores = raw_attn_scores.squeeze(dim=0) # (slot_num, dim)

            if s_attn.size(0) > 1:
                entropy_attn = s_attn.detach().clone().cpu().numpy()
                entropy_scores = entropy(entropy_attn, base=2, axis=0)
                entropy_scores = np.average(entropy_scores)
                if nsent[i] not in entropy_map:
                    entropy_map[nsent[i]] = []
                entropy_map[nsent[i]].append(entropy_scores)


            # run predicate-to-slot assignment
            attn_max = torch.max(s_attn, dim=0)[0]
            attn_max = torch.stack([attn_max] * s_attn.size(0), dim=0)
            _predicate_to_slot = (s_attn == attn_max).int()

            predicate_to_slot = []
            for j in range(_predicate_to_slot.size(0)):
                select_idx = _predicate_to_slot[j].nonzero().view(-1)
                to_slot = pred[i].index_select(0, select_idx).tolist()
                predicate_to_slot.append(to_slot)

            predicate_to_sentence = p2s[i]
            slot_to_sentence_costs = np.zeros((nsent[i], nsent[i])) # slot to sentence alignments
            for j in range(len(predicate_to_slot)):
                for k in range(len(predicate_to_sentence)):
                    overlap = len(set(predicate_to_slot[j]) & set(predicate_to_sentence[k]))*2+1
                    slot_to_sentence_costs[j][k] = overlap

            #print (predicate_to_slot, predicate_to_sentence)
            #print (slot_to_sentence_costs)
            slot_to_sentence_costs = slot_to_sentence_costs.max() - slot_to_sentence_costs
            #print (slot_to_sentence_costs)
            row_ind, col_ind = linear_sum_assignment(slot_to_sentence_costs)
            #print (row_ind, col_ind)
            #print (tgt[i])
            new_tgt = torch.stack([tgt[i][idx] for idx in col_ind])
            #print (new_tgt)
            new_tgt_mask = torch.stack([mask_tgt[i][idx] for idx in col_ind])
            #print ('\n')

            #slot_embs.append(s_embs)
            if not self.args.slot_sample_schedule:
                _attn_matrix = F.gumbel_softmax(raw_attn_scores, tau=0.2, hard=True, dim=0)
            else:
                switch = random.uniform(0, 1)
                if switch < train_progress:
                    _attn_matrix = F.gumbel_softmax(raw_attn_scores, tau=0.2, hard=True, dim=0)
                else:
                    _attn_matrix = s_attn
            new_s_embs = torch.mm(_attn_matrix, p_emb.squeeze(dim=0))
            slot_embs.append(new_s_embs)

            new_tgts.append(new_tgt)
            new_tgt_masks.append(new_tgt_mask)

        slot_embs = torch.cat(slot_embs, dim=0).unsqueeze(dim=1)
        tgt = torch.cat(new_tgts, dim=0)
        mask_tgt = torch.cat(new_tgt_masks, dim=0)

        # Run Decoder
        tgt_embs = self.decoder.embed_tokens(tgt)
        tgt_embs = torch.cat((slot_embs, tgt_embs[:,1:,:]), dim=1)
        decoder_outputs = self.decoder(inputs_embeds=tgt_embs, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src)

        return decoder_outputs.last_hidden_state, tgt, mask_tgt, entropy_map


class SlotAttnAggragatorDiscrete(nn.Module):
    def __init__(self, args, device, vocab_size, pad_id, first_sent_lable, not_first_sent_label, checkpoint=None, pretrained_checkpoint=None):
        super(SlotAttnAggragatorDiscrete, self).__init__()
        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.original_vocab_size = len(self.original_tokenizer)
        self.first_sent_lable = first_sent_lable
        self.not_first_sent_label = not_first_sent_label
        self.pad_id = pad_id

        if pretrained_checkpoint is not None:
            self.model = pretrained_checkpoint
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
            self.generator = self.model.generator
            model_dim = 768
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
            if self.vocab_size > self.original_vocab_size:
                self.model.resize_token_embeddings(self.vocab_size)
            self.encoder = self.model.get_encoder()
            self.decoder = self.model.get_decoder()
            model_dim = self.model.config.hidden_size
            self.generator = get_generator(self.vocab_size, model_dim, device)

        self.planner = SlotAttention(args.slot_num_slots, model_dim, args.slot_iters, args.slot_eps, model_dim)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            print ('Initialize parameters for generator...')
            if pretrained_checkpoint is None:
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()
            for p in self.planner.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

        if args.freeze_encoder_decoder:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.generator.parameters():
                param.requires_grad = False

        self.to(device)


    def forward(self, src, tgt, pred, p2s, mask_src, mask_tgt, nsent, train_progress=1):

        # Run encoder
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        # Run slot attn
        slot_embs = []; new_tgts = []; new_tgt_masks = []; new_loss_masks = []
        entropy_map = {}
        for i in range(len(nsent)):
            # for the i-th example, run slot attention
            p_emb = self.decoder.embed_tokens(pred[i])
            p_emb = p_emb.unsqueeze(dim=0) # (batch_size=1, pred_num, dim)
            s_embs, s_attn, raw_attn_scores = self.planner(p_emb, num_slots=nsent[i]) 
            # (batch_size=1, slot_num, dim), (batch_size=1, slot_num, pred_num)
            s_embs = s_embs.squeeze(dim=0) # (slot_num, dim)
            s_attn = s_attn.squeeze(dim=0) # (slot_num, dim)

            # debug entropy
            if s_attn.size(0) > 1:
                entropy_attn = s_attn.detach().clone().cpu().numpy()
                entropy_scores = entropy(entropy_attn, base=2, axis=0)
                entropy_scores = np.average(entropy_scores)
                if nsent[i] not in entropy_map:
                    entropy_map[nsent[i]] = []
                entropy_map[nsent[i]].append(entropy_scores)

            # run predicate-to-slot assignment
            _predicate_to_slot = F.gumbel_softmax(s_attn, tau=0.1, hard=True, dim=0) # for future use

            predicate_to_slot = []
            for j in range(_predicate_to_slot.size(0)):
                select_idx = _predicate_to_slot[j].nonzero().view(-1)
                to_slot = pred[i].index_select(0, select_idx).tolist()
                predicate_to_slot.append(to_slot)

            predicate_to_sentence = p2s[i]
            slot_to_sentence_costs = np.zeros((nsent[i], nsent[i])) # slot to sentence alignments
            for j in range(len(predicate_to_slot)):
                for k in range(len(predicate_to_sentence)):
                    overlap = len(set(predicate_to_slot[j]) & set(predicate_to_sentence[k]))*2+1
                    slot_to_sentence_costs[j][k] = overlap

            slot_to_sentence_costs = slot_to_sentence_costs.max() - slot_to_sentence_costs
            row_ind, col_ind = linear_sum_assignment(slot_to_sentence_costs)
            new_tgt = torch.stack([tgt[i][idx] for idx in col_ind])
            new_tgt_mask = torch.stack([mask_tgt[i][idx] for idx in col_ind])
            new_loss_mask = torch.zeros(new_tgt.size(), device=new_tgt.device)

            # edit slot embedding based one the sampling result
            new_s_embs = torch.mm(_predicate_to_slot, p_emb.squeeze(dim=0))
            # get sampled predicates
            for k, sent in enumerate(new_tgt):
                # keep the sentence
                for j, tok in enumerate(sent):
                    if tok == self.first_sent_lable or tok == self.not_first_sent_label:
                        sent_len = new_tgt_mask[k][j:].sum()
                        pure_sent = sent[j:j+sent_len].clone()
                        break
                # erase tgt
                new_tgt[k][1:] = self.pad_id
                # copy selected slots
                pred_list = pred[i]
                sampled_indicator = _predicate_to_slot[k]
                tgt_idx = 1
                for j in range(sampled_indicator.size(0)):
                    if sampled_indicator[j]:
                        new_tgt[k][tgt_idx] = pred_list[j]
                        tgt_idx += 1
                # copy tgt tokens
                new_tgt[k][tgt_idx:tgt_idx+pure_sent.size(0)] = pure_sent
                new_loss_mask[k][tgt_idx:tgt_idx+pure_sent.size(0)] = 1

            slot_embs.append(new_s_embs)
            new_tgts.append(new_tgt)

            new_tgt_mask = ~(new_tgt == self.pad_id)
            new_tgt_masks.append(new_tgt_mask)

            new_loss_masks.append(new_loss_mask)


        slot_embs = torch.cat(slot_embs, dim=0).unsqueeze(dim=1)
        tgt = torch.cat(new_tgts, dim=0)
        mask_tgt = torch.cat(new_tgt_masks, dim=0)
        loss_mask = torch.cat(new_loss_masks, dim=0)

        # Run Decoder
        tgt_embs = self.decoder.embed_tokens(tgt)
        tgt_embs = torch.cat((slot_embs, tgt_embs[:,1:,:]), dim=1)
        decoder_outputs = self.decoder(inputs_embeds=tgt_embs, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src)

        # loss calculation
        tgt = (tgt * loss_mask) + self.pad_id * (1-loss_mask)
        tgt = tgt.long()

        return decoder_outputs.last_hidden_state, tgt, loss_mask, entropy_map

