import copy, random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
from models.optimizers import Optimizer
from transformers import AutoTokenizer, BertModel
from models.encoder import SentenceClassification, ClusterClassification
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



class ExtSummarizer(nn.Module):
    def __init__(self, args, device, vocab_size, checkpoint, sentence_modelling_for_ext):
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
    def __init__(self, args, device, checkpoint, sentence_modelling_for_ext):
        super(ParagraphMultiClassifier, self).__init__()
        self.args = args
        self.device = device
        self.local_bert = BertModel.from_pretrained(args.model_name)
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
        token_embeddings = self.local_bert(input_ids=src, attention_mask=mask_src)
        res = self.sentence_pooling({'token_embeddings':token_embeddings.last_hidden_state, 'attention_mask':mask_src})
        # number of sentences * embedding size 
        sentence_embeddings = res['sentence_embedding']
        # number of clusters * embedding size 
        cluster_embeddings = self.pool_cluster_embeddings(sentence_embeddings, cluster_sizes)
        verd_scores, pros_scores, cons_scores, example_masks = self.cluster_classification(cluster_embeddings, cluster_sizes)

        return verd_scores, pros_scores, cons_scores, example_masks
