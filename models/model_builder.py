import copy, random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from models.optimizers import Optimizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import LongformerModel, BertModel
from models.encoder import SentenceClassification, ClusterClassification, TransformerEncoder, Classifier, PairClassification
from models.slot_attn import SlotAttention, SoftKMeans
from models.spectral_clustering import SpectralCluser
from sentence_transformers.models import Pooling

def build_optim(args, model, checkpoint, lr=None, warmup_steps=None):
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
    def __init__(self, args, device, vocab_size, checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = vocab_size

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

            if not self.args.slot_sample_schedule:
                #_attn_matrix = F.gumbel_softmax(raw_attn_scores, tau=0.2, hard=True, dim=0)
                _attn_matrix = s_attn
            else:
                switch = random.uniform(0, 1)
                if switch < train_progress:
                    _attn_matrix = F.gumbel_softmax(raw_attn_scores, tau=0.2, hard=True, dim=0)
                else:
                    _attn_matrix = s_attn

            print (p2s[i], pred[i])
            print (_attn_matrix)
            print (new_tgt, new_tgt_mask)
            print ('\n\n')
            p_emb_dec = self.decoder.embed_tokens(pred[i])
            new_s_embs = torch.mm(_attn_matrix, p_emb_dec.squeeze(dim=0))
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

        if self.args.cluster_algorithm == 'soft_kmeans':
            self.planner = SoftKMeans(args.slot_num_slots, model_dim, args.slot_iters, args.slot_eps, model_dim)
        else:
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


class SlotSumm(nn.Module):
    def __init__(self, args, device, vocab_size, pad_token_id, checkpoint=None):
        super(SlotSumm, self).__init__()
        self.args = args
        self.device = device
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.train_stage = self.args.slotsumm_train_stage

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.token_encoder = self.model.get_encoder()
        self.sentence_encoder = copy.deepcopy(self.token_encoder)
        self.decoder = self.model.get_decoder()

        model_dim = self.model.config.hidden_size
        self.generator = get_generator(self.vocab_size, model_dim, device)

        if self.args.cluster_algorithm == 'soft_kmeans':
            self.planner = SoftKMeans(args.slot_num_slots, model_dim, args.slot_iters, args.slot_eps, hidden_dim=model_dim)
        else:
            self.planner = SlotAttention(args.slot_num_slots, model_dim, args.slot_iters, args.slot_eps, hidden_dim=model_dim)

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
            for p in self.planner.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()

        if args.freeze_encoder_decoder:
            for param in self.token_encoder.parameters():
                param.requires_grad = False
            for param in self.sentence_encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.to(device)

    def slot_attn(self, tgt, tgt_mask, s2t, sentence_emb, num_slots):
        print (tgt.size(), tgt_mask.size(), sentence_emb.size(), s2t)
        p_emb = sentence_emb
        p_emb = p_emb.unsqueeze(dim=0) # (batch_size=1, nsent_src, dim)
        s_embs, s_attn, raw_attn_scores = self.planner(p_emb, num_slots=num_slots) # (batch_size=1, slot_num, dim), (batch_size=1, slot_num, pred_num)
        s_embs = s_embs.squeeze(dim=0) # (slot_num, dim)
        s_attn = s_attn.squeeze(dim=0) # (slot_num, dim)
        raw_attn_scores = raw_attn_scores.squeeze(dim=0) # (slot_num, dim)

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

        if not self.args.slot_sample_schedule:
            #_attn_matrix = F.gumbel_softmax(raw_attn_scores, tau=0.2, hard=True, dim=0)
            _attn_matrix = s_attn
        else:
            switch = random.uniform(0, 1)
            if switch < train_progress:
                _attn_matrix = F.gumbel_softmax(raw_attn_scores, tau=0.2, hard=True, dim=0)
            else:
                _attn_matrix = s_attn

        print (p2s[i], pred[i])
        print (_attn_matrix)
        print (new_tgt, new_tgt_mask)
        print ('\n\n')


    def ground_truth_attn(self, src_to_tgt, src_embeddings):
        nsent_src = src_embeddings.size(0)
        nsent_tgt = len(src_to_tgt)
        gold_attn = torch.zeros(nsent_tgt, nsent_src, device=self.device)
        for i in range(len(src_to_tgt)):
            for idx in src_to_tgt[i]:
                gold_attn[i][idx] = 1
        return gold_attn


    def forward(self, src, src_sents, tgt, tgt_sents, mask_src, mask_src_sents, mask_tgt, mask_tgt_sents, nsent_src, s2t, run_decoder=True):

        # Run token-level encoder
        token_encoder_outputs = self.token_encoder(input_ids=src_sents, attention_mask=mask_src_sents) 
        sentence_emb = token_encoder_outputs.last_hidden_state[:, 0, :]

        # Run sentence-level encoder
        idx = 0
        width = max(nsent_src)
        nonpad_sentence_emb = []; pad_sentence_emb = []; sentence_mask = []
        for i, nsent in enumerate(nsent_src):
            example_src_emb = sentence_emb[idx:idx+nsent]
            nonpad_sentence_emb.append(sentence_emb[idx:idx+nsent])
            if width-nsent == 0:
                pad_sentence_emb.append(example_src_emb)
            else:
                pads = torch.tensor([self.pad_token_id] * (width-nsent), device=self.device).int()
                pad_embeddings = self.sentence_encoder.embed_tokens(pads)
                pad_sentence_emb.append(torch.cat(example_src_emb, pad_embeddings))
            smask = [True] * nsent + [False] * (width-nsent)
            sentence_mask.append(smask)
            idx += nsent
        pad_sentence_emb = torch.stack(pad_sentence_emb, dim=0) # n_example * n_sent * dim
        sentence_mask = torch.tensor(sentence_mask, device=self.device)
        sentence_encoder_outputs = self.sentence_encoder(inputs_embeds=pad_sentence_emb, attention_mask=sentence_mask)
        top_vec = sentence_encoder_outputs.last_hidden_state

        if not run_decoder:
            return {"encoder_outpus":top_vec, "encoder_attention_mask":sentence_mask}

        # slotsumm_train_stage is pre-train
        if self.train_stage == 'pre-train':
            decoder_outputs = self.decoder(input_ids=tgt, 
                                           attention_mask=mask_tgt,
                                           encoder_hidden_states=top_vec,
                                           encoder_attention_mask=sentence_mask)
            return decoder_outputs.last_hidden_state, tgt, mask_tgt, None

        # Run slot attn
        slot_embs = []; new_tgts = []; new_tgt_masks = []; entropy_map = {}
        new_top_vec = []; new_sentence_masks = []
        nsent_tgt = [len(item) for item in s2t]; row_idx = 0
        for i in range(len(s2t)):
            # for the i-th example, run slot attention
            if self.train_stage == 'gold_align':
                _attn_matrix = self.ground_truth_attn(s2t[i], nonpad_sentence_emb[i])
                example_tgt = tgt_sents[row_idx:row_idx+nsent_tgt[i]]
                example_tgt_mask = mask_tgt_sents[row_idx:row_idx+nsent_tgt[i]]
                row_idx += nsent_tgt[i]
            else:
                # self.train_stage == 'slot_attn'
                example_tgt = tgt_sents[row_idx:row_idx+nsent_tgt[i]]
                example_tgt_mask = mask_tgt_sents[row_idx:row_idx+nsent_tgt[i]]
                _attn_matrix, example_tgt, example_tgt_mask = self.slot_attn(example_tgt, 
                                                                             example_tgt_mask, 
                                                                             s2t[i], 
                                                                             nonpad_sentence_emb[i],
                                                                             nsent_tgt[i])
                row_idx += nsent_tgt[i]

            # calculate slot_embs
            example_slot_embs = torch.mm(_attn_matrix, nonpad_sentence_emb[i])
            slot_embs.append(example_slot_embs)
            new_tgts.append(example_tgt)
            new_tgt_masks.append(example_tgt_mask)

            # duplicate sentence_encoder output
            example_src_embedding = top_vec[i].unsqueeze(dim=0)
            example_src_embedding = example_src_embedding.repeat(nsent_tgt[i], 1, 1)
            new_top_vec.append(example_src_embedding)

            # duplicate sentence_mask
            example_sentence_mask = sentence_mask[i].unsqueeze(dim=0)
            example_sentence_mask = example_sentence_mask.repeat(nsent_tgt[i], 1)
            new_sentence_masks.append(example_sentence_mask)

        slot_embs = torch.cat(slot_embs, dim=0).unsqueeze(dim=1)
        tgt = torch.cat(new_tgts, dim=0)
        mask_tgt = torch.cat(new_tgt_masks, dim=0)
        top_vec = torch.cat(new_top_vec, dim=0)
        sentence_mask = torch.cat(new_sentence_masks, dim=0)

        # Run Decoder
        tgt_embs = self.decoder.embed_tokens(tgt)
        tgt_embs = torch.cat((slot_embs, tgt_embs[:,1:,:]), dim=1)
        decoder_outputs = self.decoder(inputs_embeds=tgt_embs, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=sentence_mask)

        return decoder_outputs.last_hidden_state, tgt, mask_tgt, entropy_map



class SpectralReinforce(nn.Module):
    def __init__(self, args, device, pad_id, vocab_size, tokenizer, abs_model=None, checkpoint=None):
        super(SpectralReinforce, self).__init__()
        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.warmup_steps_reinforce = args.warmup_steps_reinforce

        self.tokenizer = tokenizer

        if abs_model is not None:
            self.abs_model = abs_model
        else:
            self.abs_model = AbsSummarizer(args, device, vocab_size)

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
        self.nll = nn.NLLLoss(ignore_index=self.pad_id, reduce=False)
        self.cls_loss = torch.nn.BCELoss(reduction='none')

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


    def run_spectral(self, predicates, n_clusters, src_str=None, pred_str=None):
        if len(predicates) == 1:
            return [0]
        ajacency_matrix = torch.zeros(len(predicates),len(predicates), device=self.device)
        for i, head_1 in enumerate(predicates):
            for j, head_2 in enumerate(predicates):
                if head_1 < head_2:
                    ajacency_matrix[i][j] = self.sigmoid(self.predicate_graph[head_1][head_2])
                else:
                    ajacency_matrix[i][j] = self.sigmoid(self.predicate_graph[head_2][head_1])
                if self.args.test_entity_link and \
                        src_str is not None and \
                        (not self._examine_src_pairs(src_str[i], src_str[j])):
                    ajacency_matrix[i][j] = 0
        '''
        print (ajacency_matrix)
        print (predicates)
        ajacency_matrix_sample = torch.bernoulli(ajacency_matrix)
        ajacency_matrix_sample_t = torch.transpose(ajacency_matrix_sample, 0, 1)
        ajacency_matrix_sample = ajacency_matrix_sample + ajacency_matrix_sample_t
        ajacency_matrix_sample = (ajacency_matrix_sample > 0).int()
        #ajacency_matrix = (ajacency_matrix * ajacency_matrix_sample).cpu().detach().numpy()
        ajacency_matrix = ajacency_matrix_sample.cpu().detach().numpy()
        '''

        ajacency_matrix = ajacency_matrix.cpu().detach().numpy()
        print (n_clusters)
        print (pred_str)
        print (ajacency_matrix)
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels='discretize',
                                        eigen_solver='arpack',
                                        affinity='precomputed').fit(ajacency_matrix)
        print (clustering.labels_)
        return clustering.labels_


    def run_spectral_nn(self, predicates, pred_token, pred_token_mask, n_clusters, src_str=None, pred_str=None):

        if len(predicates) == 1:
            return [0]

        linear_ajacency_matrix = self.predicate_graph(pred_token, pred_token_mask)
        #print (linear_ajacency_matrix.view(len(predicates), -1))
        if self.args.spectral_with_sample:
            linear_ajacency_matrix = torch.bernoulli(linear_ajacency_matrix)

        sub_graph = {}; idx = 0
        for i, head_i in enumerate(predicates):
            sub_graph[head_i] = {}
            for j, head_j in enumerate(predicates):
                sub_graph[head_i][head_j] = linear_ajacency_matrix[idx]
                idx += 1

        ajacency_matrix = linear_ajacency_matrix.view(len(predicates), -1)
        for i, head_i in enumerate(predicates):
            for j, head_j in enumerate(predicates):
                if head_i < head_j:
                    ajacency_matrix[i][j] = sub_graph[head_i][head_j]
                elif head_i > head_j:
                    ajacency_matrix[i][j] = sub_graph[head_j][head_i]

        for i, head_1 in enumerate(predicates):
            for j, head_2 in enumerate(predicates):
                if self.args.test_entity_link and \
                        src_str is not None and \
                        (not self._examine_src_pairs(src_str[i], src_str[j])):
                    ajacency_matrix[i][j] = 0

        ajacency_matrix = ajacency_matrix.cpu().detach().numpy()

        #print ('NN', n_clusters)
        #print (pred_str)
        #print (ajacency_matrix)
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels='discretize',
                                        eigen_solver='arpack',
                                        affinity='precomputed').fit(ajacency_matrix)
        #print (clustering.labels_)
        return clustering.labels_


    def run_random(self, predicates, n_clusters):
        labels = [random.sample(range(n_clusters), 1)[0] for item in predicates]
        return labels


    def run_discriministic(self, src, preds, n_clusters):
        labels = self.deterministic_graph.get_aggragation_lable(preds, src, n_clusters)
        return labels


    def calculate_graph_prob(self, pred_groups):
        # TODO: carefully design is required
        probs = []
        for group in pred_groups:
            log_prob = []
            for i, head_1 in enumerate(group):
                for j, head_2 in enumerate(group):
                    if head_1 < head_2:
                        likelihood = self.sigmoid(self.predicate_graph[head_1][head_2])
                        #print (head_1, head_2, likelihood)
                        log_likelihood = torch.log(likelihood)
                        log_prob.append(log_likelihood)
                    elif head_2 < head_1:
                        likelihood = self.sigmoid(self.predicate_graph[head_2][head_1])
                        #print (head_1, head_2, likelihood)
                        log_likelihood = torch.log(likelihood)
                        log_prob.append(log_likelihood)
                    elif len(group) == 1:
                        likelihood = self.sigmoid(self.predicate_graph[head_2][head_1])
                        if self.args.test_no_single_pred_score:
                            likelihood = torch.tensor(1.0, device=self.device)
                        #print (head_1, head_2, likelihood)
                        log_likelihood = torch.log(likelihood)
                        log_prob.append(log_likelihood)
            if len(log_prob) > 0:
                if self.args.calculate_graph_prob_method == 'min':
                    log_prob = min(log_prob)
                else:
                    log_prob = sum(log_prob)/len(log_prob)
                probs.append(log_prob)
        return probs


    def calculate_graph_prob_nn(self, pred_groups, predicates, pred_token, pred_token_mask):

        # TODO: carefully design is required
        linear_ajacency_matrix = self.predicate_graph(pred_token, pred_token_mask)
        sub_graph = {}; idx = 0
        for i, head_i in enumerate(predicates):
            sub_graph[head_i] = {}
            for j, head_j in enumerate(predicates):
                sub_graph[head_i][head_j] = linear_ajacency_matrix[idx]
                idx += 1

        probs = []
        for group in pred_groups:
            log_prob = []
            for i, head_1 in enumerate(group):
                for j, head_2 in enumerate(group):
                    if head_1 < head_2:
                        likelihood = sub_graph[head_1][head_2]
                        #print (head_1, head_2, likelihood)
                        log_likelihood = torch.log(likelihood)
                        log_prob.append(log_likelihood)
                    elif head_1 > head_2:
                        likelihood = sub_graph[head_2][head_1]
                        #print (head_1, head_2, likelihood)
                        log_likelihood = torch.log(likelihood)
                        log_prob.append(log_likelihood)
                    elif len(group) == 1:
                        likelihood = sub_graph[head_1][head_2]
                        if self.args.test_no_single_pred_score:
                            likelihood = torch.tensor(1.0, device=self.device)
                        log_likelihood = torch.log(likelihood)
                        log_prob.append(log_likelihood)
            if len(log_prob) > 0:
                if self.args.calculate_graph_prob_method == 'min':
                    log_prob = min(log_prob)
                else:
                    log_prob = sum(log_prob)/len(log_prob)
                probs.append(log_prob)
        return probs


    def run_clustering(self, src, preds, n_clusters, p2s, pred_token, pred_token_mask, mode='spectral', src_str=None, pred_str=None):
        
        # run clustering
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
                select_pred = random.sample(range(len(preds)), 1)[0]
                labels = [-1] * len(labels)
                labels[select_pred] = 0

        elif mode == 'discriministic':
            labels = self.run_discriministic(src_str, pred_str, n_clusters)
        else:
            if self.args.nn_graph:
                labels = self.run_spectral_nn(preds, pred_token, pred_token_mask, n_clusters, src_str, pred_str)
            else:
                labels = self.run_spectral(preds, n_clusters, src_str, pred_str)

        # group predicates and src based on the cluster method
        pred_groups = [[] for i in range(n_clusters)]
        tmp_src_groups = [[] for i in range(n_clusters)]
        for i, label in enumerate(labels):
            if label == -1:
                continue
            pred_groups[label].append(preds[i])
            tmp_src_groups[label].append(src[i])

        # for test start
        pred_str_groups = None; src_str_groups = None
        if (src_str is not None) and (pred_str is not None):
            pred_str_groups = [[] for i in range(n_clusters)]
            src_str_groups = [[] for i in range(n_clusters)]
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                pred_str_groups[label].append(pred_str[i])
                src_str_groups[label].append(src_str[i])
        # for test end

        src_groups = [[] for i in range(n_clusters)]
        for i in range(n_clusters):
            sg = tmp_src_groups[i]
            if self.args.shuffle_src:
                random.shuffle(sg)
            for item in sg:
                src_groups[i].extend(item)

        if mode == 'discriministic':
            graph_prob = self.deterministic_graph.calculate_graph_score(labels, pred_str, n_clusters)
        else:
            if self.args.nn_graph:
                graph_prob = self.calculate_graph_prob_nn(pred_groups, preds, pred_token, pred_token_mask)
            else:
                graph_prob = self.calculate_graph_prob(pred_groups)

        return src_groups, pred_groups, graph_prob, src_str_groups, pred_str_groups


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

            # run clustering
            src_groups, pred_groups, graph_probs, _, _ = self.run_clustering(s, p, n_clusters, p_s, p_tok, p_tok_m, mode=mode)

            # run cluster-to-output_sentence alignment
            res = self.hungrian_alignment(pred_groups, t, m_t, ct, m_ct, m_ct_l, p_s, n_clusters)
            new_tgt, new_tgt_mask, new_ctgt, new_ctgt_mask, new_mask_ctgt_loss = res

            parallel_src.extend(src_groups)
            parallel_tgt.append(new_tgt)
            parallel_tgt_mask.append(new_tgt_mask)
            parallel_ctgt.append(new_ctgt)
            parallel_ctgt_mask.append(new_ctgt_mask)
            parallel_tgt_mask_loss.append(new_mask_ctgt_loss)
            parallel_graph_probs.extend(graph_probs)
            ngroups.append(t.size(0))

        src = torch.tensor(self._pad(parallel_src), device=self.device)
        mask_src = ~(src == self.pad_id)
        encoder_outputs = self.abs_model.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        if not run_decoder:
            return {"encoder_outpus":top_vec, "encoder_attention_mask":mask_src}

        # Decoding
        tgt = torch.cat(parallel_tgt)
        mask_tgt = torch.cat(parallel_tgt_mask)
        ctgt = torch.cat(parallel_ctgt)
        mask_ctgt = torch.cat(parallel_ctgt_mask)
        mask_ctgt_loss = torch.cat(parallel_tgt_mask_loss)
        gtruth = tgt[:, 1:].contiguous().view(-1)

        if self.args.conditional_decoder:
            tgt = ctgt
            mask_tgt = mask_ctgt
            gtruth = tgt * mask_ctgt_loss + self.pad_id * (~ mask_ctgt_loss)
            gtruth = gtruth[:, 1:].contiguous().view(-1)

        '''
        for i in range(src.size(0)):
            print ('[SRC]:'+' '.join(self.tokenizer.convert_ids_to_tokens(src[i])))
            print ('[TGT]:'+' '.join(self.tokenizer.convert_ids_to_tokens(tgt[i])))
            gt = gtruth.view(src.size(0), -1)
            print ('[GT]:'+' '.join(self.tokenizer.convert_ids_to_tokens(gt[i])))
            print ('\n')
        print ('=================')
        '''

        decoder_outputs = self.abs_model.decoder(input_ids=tgt, 
                                                attention_mask=mask_tgt,
                                                encoder_hidden_states=top_vec,
                                                encoder_attention_mask=mask_src)

        output = decoder_outputs.last_hidden_state
        output = output[:, :-1, :]

        bottled_output = output.reshape(-1, output.size(2))
        scores = self.abs_model.generator(bottled_output)
        log_likelihood = (-1) * self.nll(scores, gtruth)
        logging_info = self._log_generation_stats(scores, gtruth, src) # log_info

        '''
        # log
        for i in range(output.size(0)):
            gtruth = gtruth.view(src.size(0), -1)
            print (' '.join(self.tokenizer.convert_ids_to_tokens(src[i])), '|||||||',  ' '.join(self.tokenizer.convert_ids_to_tokens(gtruth[i])))
        '''

        if self.args.pretrain_encoder_decoder:
            return log_likelihood, None, logging_info


        log_likelihood = log_likelihood.view(tgt.size(0),-1)
        log_likelihood = torch.mean(log_likelihood, dim=1)

        weights = torch.stack(parallel_graph_probs)

        # post process
        cur_id = 0
        processed_log_likelihood = []
        processed_weights = []
        for nline in nsent:
            processed_log_likelihood.append(log_likelihood[cur_id:(cur_id+nline)].min())
            processed_weights.append(weights[cur_id:(cur_id+nline)].sum())
            cur_id += nline

        log_likelihood = torch.stack(processed_log_likelihood)
        weights = torch.stack(processed_weights)

        '''
        # log
        print (log_likelihood)
        print ('\n')
        '''

        return log_likelihood, weights, logging_info


    def forward_cls(self, pred_tokens, pred_mask_tokens, aggregation_labels, nsents):

        tokens = torch.cat(pred_tokens)
        masks = torch.cat(pred_mask_tokens)
        labels = torch.cat([item.unsqueeze(1) for item in aggregation_labels]).squeeze(-1)

        sent_scores = self.predicate_graph(tokens, masks)
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
