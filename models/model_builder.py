import copy, random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
from models.encoder import Classifier, TreeInference, SentenceClassification
from models.decoder import BartDecoderCS
from models.t5_encoder_decoder import T5Stacker
from models.optimizers import Optimizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from models.tree_reader import tree_to_content_mask, tree_building, gumbel_softmax_function, topn_function
from models.neural import SimpleSelfAttention

def build_optim_enc_dec(args, model, checkpoint):
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
            args.optim, args.lr_enc_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_enc_dec)

    params = []
    for name, para in model.named_parameters():
        if not name.startswith('planning_layer'):
            params.append((name, para))
    optim.set_parameters(params)
    return optim


def build_optim_tmt(args, model, checkpoint):
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
            args.optim, args.lr_tmt, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_tmt)

    params = []
    for name, para in model.named_parameters():
        if name.startswith('planning_layer'):
            print (name)
            params.append((name, para))
    optim.set_parameters(params)
    return optim


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
        print (self.vocab_size, len(self.original_tokenizer))
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        if sentence_modelling_for_ext == 'tree':
            self.planning_layer = TreeInference(self.model.config.hidden_size, 
                                                args.ext_ff_size, 
                                                args.ext_dropout, 
                                                args.ext_layers)
        else:
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
        self.gumbel_tau = args.gumbel_tau

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


    def forward(self, src, tgt, mask_src, mask_tgt, 
                mask_src_sent=None, mask_tgt_sent=None, 
                tgt_nsent=None, mask_src_predicate=None,
                prompt_tokenized=None, alignments=None,
                run_decoder=True):

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



class StepAbsSummarizer(nn.Module):
    def __init__(self, args, device, cls_token_id, vocab_size, checkpoint=None, abs_finetune=None):
        super(StepAbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = vocab_size
        self.cls_token_id = cls_token_id
        self.gumbel_tau = args.gumbel_tau
        self.softmax = nn.Softmax(dim=-1)

        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        self.decoder = T5Stacker(decoder)
        self.generator = get_generator(self.vocab_size, self.model.config.hidden_size, device)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            if abs_finetune is not None:
                print ('Load parameters from abs_finetune...')
                tree_params = [(n[8:], p) for n, p in abs_finetune['model'].items() if n.startswith('encoder')]
                self.encoder.load_state_dict(dict(tree_params), strict=True)
                tree_params = [(n[8:], p) for n, p in abs_finetune['model'].items() if n.startswith('decoder')]
                self.decoder.load_state_dict(dict(tree_params), strict=True)
                tree_params = [(n[10:], p) for n, p in abs_finetune['model'].items() if n.startswith('generator')]
                self.generator.load_state_dict(dict(tree_params), strict=True)
            else:
                print ('Initialize parameters for generator...')
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()

        self.to(device)

    def predicate_self_attention(self, encoder_outputs, src_mask, alignments, predicate_idx):
        key = encoder_outputs
        query = encoder_outputs
        query = query / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(1, 2))
        src_mask = src_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~src_mask, -1e9)
        attn = scores
        #attn = self.softmax(scores)
        step_weights = []
        for example_id in range(attn.size(0)):
            pred_idx = predicate_idx[example_id]
            example_attn = attn[example_id]
            predicate_attns = torch.index_select(example_attn, 0, pred_idx)
            alignment = alignments[example_id]
            step_weight = []
            for group in alignment:
                pred_attn_max_pool = torch.max(predicate_attns[group], dim=0)[0]
                step_weight.append(pred_attn_max_pool)
            step_weights.append(step_weight)
        return step_weights

    def step_weights_expand(self, step_weights, mask_tgt_sent):
        device = mask_tgt_sent.device
        tgt_len = mask_tgt_sent.size(-1)
        cross_attn_mask = []
        for i, step_weight in enumerate(step_weights):
            tgt_sent = mask_tgt_sent[i]
            # for each sent in tgt
            c_mask = []
            for j, sent_weight in enumerate(step_weight):
                sent_mask = sent_weight.unsqueeze(0)
                sent_mask = sent_mask.repeat((int(tgt_sent[j].sum()), 1))
                c_mask.append(sent_mask)
            c_mask = torch.cat(c_mask)
            # padding
            mask_padding = torch.zeros((tgt_len-c_mask.size(0), c_mask.size(1)), device=device)
            #print (c_mask.size(), mask_padding.size())
            c_mask = torch.cat([c_mask, mask_padding])
            cross_attn_mask.append(c_mask)
        return torch.stack(cross_attn_mask)

    def forward(self, src, tgt, mask_src, mask_tgt, 
                mask_src_sent, mask_tgt_sent, 
                alignments, src_predicate_token_idx, 
                run_decoder=True):

        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        if not run_decoder:
            return {"encoder_outpus":encoder_outputs.last_hidden_state, "encoder_attention_mask":mask_src}

        if self.args.cross_attn_weight_format == 'pred_selfattn':
            step_weights = self.predicate_self_attention(top_vec, mask_src, alignments, src_predicate_token_idx)
            content_selection_weights = self.step_weights_expand(step_weights, mask_tgt_sent)
        else:
            content_selection_weights = tree_to_content_mask(alignments, mask_src_sent, mask_tgt_sent)
        content_selection_weights = {'weights':content_selection_weights, 'format':self.args.cross_attn_weight_format}

        # Decoding
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src,
                                       content_weights=content_selection_weights)

        return decoder_outputs.last_hidden_state



class SoftSrcPromptSummarizer(nn.Module):
    def __init__(self, args, device, tokenizer, checkpoint=None):
        super(SoftSrcPromptSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = len(tokenizer)
        self.cls_token_id = tokenizer.cls_token_id

        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, self.model.config.hidden_size, device)

        special_token_ids = tokenizer.convert_tokens_to_ids(['|', '|||', '-Pred-ROOT'])
        self.skip_token_id = special_token_ids[0]
        self.seg_label_id = special_token_ids[1]
        self.position_id = special_token_ids[2]

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

    def get_prompt_segments(self, prompt_tokenized):
        prompt_segs = []
        for prompt in prompt_tokenized:
            prompt_seg = []; tmp = []
            for tok in prompt:
                if tok == self.skip_token_id:
                    continue
                if tok == self.seg_label_id:
                    prompt_seg.append(torch.stack(tmp))
                    tmp = []
                    continue
                tmp.append(tok)
            if len(tmp) > 0:
                prompt_seg.append(torch.stack(tmp))
            prompt_segs.append(prompt_seg)
        return prompt_segs

    def weighted_pooling(self, group_embs):
        element_num = group_embs.size(0)
        pooled_emb = (group_embs * (1.0/element_num)).sum(dim=0)
        return pooled_emb

    def get_embedding_with_soft_prompt(self, src, prompt_segs):
        tokens_embs = self.encoder.embed_tokens(src)
        for i, example in enumerate(prompt_segs):
            prompt_emb_list = []
            for group in example:
                prompt_group_emb = self.encoder.embed_tokens(group)
                prompt_emb_list.append(self.weighted_pooling(prompt_group_emb))
            position_ids = (src[i] == self.position_id).nonzero(as_tuple=True)[0]
            for j, position_id in enumerate(position_ids):
                tokens_embs[i][position_id] = prompt_emb_list[j]
        return tokens_embs

    def forward(self, src, tgt, mask_src, mask_tgt, 
                mask_src_sent=None, mask_tgt_sent=None, 
                tgt_nsent=None, mask_src_predicate=None,
                prompt_tokenized=None, alignments=None,
                run_decoder=True):

        prompt_segs = self.get_prompt_segments(prompt_tokenized)
        input_embs = self.get_embedding_with_soft_prompt(src, prompt_segs)

        #encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        encoder_outputs = self.encoder(inputs_embeds=input_embs, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state
        content_selection_weights = mask_src

        if not run_decoder:
            return {"encoder_outpus":top_vec, "encoder_attention_mask":content_selection_weights}

        # Decoding
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=content_selection_weights)

        return decoder_outputs.last_hidden_state



class ExtAbsSummarizer(nn.Module):

    def __init__(self, args, device, cls_token_id, checkpoint=None, ext_finetune=None, abs_finetune=None):
        super(ExtAbsSummarizer, self).__init__(args, device, cls_token_id, checkpoint, ext_finetune, abs_finetune)
        self.args = args
        self.device = device
        model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = model.config.vocab_size
        self.cls_token_id = cls_token_id
        self.gumbel_tau = args.gumbel_tau

        # Encoder for Generator
        self.encoder = model.get_encoder()

        # Decoder for Generator
        self.decoder = model.get_decoder()
        self.generator = get_generator(self.vocab_size, model.config.hidden_size, device)

        # Planner (parameters are initialized)
        self.planning_layer = ExtSummarizer(args, device, ext_finetune, args.sentence_modelling_for_ext)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if abs_finetune is not None:
                print ('Load parameters from abs_finetune...')
                tree_params = [(n[8:], p) for n, p in abs_finetune['model'].items() if n.startswith('encoder')]
                self.encoder.load_state_dict(dict(tree_params), strict=True)
                tree_params = [(n[8:], p) for n, p in abs_finetune['model'].items() if n.startswith('decoder')]
                self.decoder.load_state_dict(dict(tree_params), strict=True)
                tree_params = [(n[10:], p) for n, p in abs_finetune['model'].items() if n.startswith('generator')]
                self.generator.load_state_dict(dict(tree_params), strict=True)
            else:
                print ('Initialize parameters for generator...')
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()

        self.decoder = BartDecoderCS(self.decoder)

        if args.freeze_encoder_decoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
        if args.freeze_tmt:
            for param in self.planning_layer.parameters():
                param.requires_grad = False

        self.to(device)


    def from_tree_to_mask(self, roots, input_ids, attention_mask, mask_block, gt_selection):
        #root_probs = torch.sum(torch.stack(roots), 0)/len(roots)
        root_probs = roots[-1]
        root_probs = root_probs * mask_block
        if self.args.planning_method == 'ground_truth':
            root_probs = gt_selection
        elif self.args.planning_method == 'topk_tree':
            root_probs = topn_function(root_probs, mask_block, self.args.ext_topn)
        elif self.args.planning_method == 'lead_k':
            root_probs = torch.zeros(gt_selection.size(), device=self.device).int()
            root_probs[:, :min(gt_selection.size(1), int(self.args.ext_topn))] = 1
            root_probs = root_probs * mask_block
        elif self.args.planning_method == 'not_lead_k':
            root_probs = torch.zeros(gt_selection.size(), device=self.device).int()
            root_probs[:, min(gt_selection.size(1), int(self.args.ext_topn)):] = 1
            root_probs = root_probs * mask_block
        elif self.args.planning_method == 'random':
            root_probs = torch.zeros(gt_selection.size(), device=self.device).int()
            nsents = mask_block.sum(dim=1)
            for i in range(root_probs.size(0)):
                sample_num = min(nsents[i], int(self.args.ext_topn))
                for idx in random.sample(range(0, int(nsents[i])), sample_num):
                    root_probs[i][idx] = 1
            root_probs = root_probs * mask_block
        else:
            root_probs = gumbel_softmax_function(root_probs, self.gumbel_tau, self.args.ext_topn)

        sep_id = self.cls_token_id
        batch_size, ntokens = input_ids.size()
        content_selection_weights = []
        for i in range(batch_size):
            example_ids = input_ids[i, :]
            sep_indices = (example_ids == sep_id).nonzero(as_tuple=True)[0].tolist()
            sep_indices[0] = 0
            sep_indices.append(input_ids.size(1))
            weights = root_probs[i, :]
            content_selection = []
            for j in range(len(sep_indices)-1):
                content_selection.append(weights[j].repeat(sep_indices[j+1]-sep_indices[j]))
            content_selection_weights.append(torch.cat(content_selection))
        content_selection_weights = torch.stack(content_selection_weights)
        content_selection_weights = content_selection_weights * attention_mask
        return content_selection_weights, root_probs


    def forward(self, src, tgt, mask_src, mask_tgt, clss, mask_cls, gt_selection, run_decoder=True):

        # Planner
        sent_scores_layers, mask_cls, aj_matrixes = self.planning_layer(src, tgt, mask_src, mask_tgt, clss, mask_cls, gt_selection)
        content_selection_weights, root_probs = self.from_tree_to_mask(sent_scores_layers, src, mask_src, mask_cls, gt_selection)

        # Encoder for Generator
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        if not run_decoder:
            return {"encoder_outpus":top_vec, 
                    "encoder_attention_mask":content_selection_weights, 
                    "sent_probs":root_probs,
                    "sent_relations":aj_matrixes[0]}

        # Decoding for Generator
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src,
                                       content_selection_mask=content_selection_weights)

        return decoder_outputs.last_hidden_state, sent_scores_layers



def _get_sentence_maxpool(top_vec, mask_src_sent, maxpool_linear):
    top_vec = top_vec.unsqueeze(1).repeat((1, mask_src_sent.size(1), 1, 1))
    mask_src_sent = mask_src_sent.unsqueeze(3).repeat((1, 1, 1, top_vec.size(-1)))
    top_vec = top_vec.masked_fill(~mask_src_sent.bool(), -1e18)
    sents_vec = torch.max(top_vec, -2)[0]
    return sents_vec


def _get_sentence_meanpool(top_vec, mask_src_sent):
    top_vec = top_vec.unsqueeze(1).repeat((1, mask_src_sent.size(1), 1, 1))
    mask_src_sent = mask_src_sent.unsqueeze(3).repeat((1, 1, 1, top_vec.size(-1)))
    top_vec = top_vec * mask_src_sent.float()
    return torch.sum(top_vec, -2)/torch.clamp(mask_src_sent.sum(-2), min=1e-9)


def _get_predicate_embedding(top_vec, mask_src_predicate, mask_cls, predicate_linear=None):
    predicates = mask_src_predicate.sum(dim=1)
    predicate_idx = [predicates[i].nonzero(as_tuple=True)[0].tolist() for i in range(predicates.size(0))]
    width = mask_cls.size(1)
    predicate_idx = [d + [-1] * (width - len(d)) for d in predicate_idx]
    sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), predicate_idx]
    if predicate_linear is not None:
        sents_vec = predicate_linear(sents_vec)
    return sents_vec


class MarginalProjectiveTreeSumm(nn.Module):

    def __init__(self, args, device, tokenizer, vocab_size, checkpoint=None, abs_finetune=None):
        super(MarginalProjectiveTreeSumm, self).__init__()

        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.cls_token_id = tokenizer.cls_token_id
        self.gumbel_tau = args.gumbel_tau
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)

        self.original_tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        print (self.vocab_size, len(self.original_tokenizer))
        if self.vocab_size > len(self.original_tokenizer):
            self.model.resize_token_embeddings(self.vocab_size)

        # Encoder for Generator
        self.encoder = self.model.get_encoder()
        # Decoder for Generator
        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, self.model.config.hidden_size, device)

        # Tree inference
        self.softmax = nn.Softmax(dim=-1)
        if args.planning_method == 'ground_truth':
            self.planning_layer = None
        elif args.planning_method == 'self_attn':
            self.planning_layer = SimpleSelfAttention(self.model.config.hidden_size)
        else:
            self.planning_layer = TreeInference(self.model.config.hidden_size, 
                                                args.ext_ff_size, 
                                                args.ext_dropout, 
                                                args.ext_layers)

        # Sentence embedding
        self.maxpool_linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True)
        self.predicate_linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, bias=True)

        # Tree embedding
        self.tree_info_wr = nn.Linear(self.model.config.hidden_size*3, self.args.tree_info_dim, bias=True)
        self.tree_info_layer_norm = nn.LayerNorm(self.model.config.hidden_size, eps=1e-6)
        self.tree_info_tanh = nn.Tanh()
        self.root_and_end_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(['-Pred-ROOT', '-Pred-END'])).int().to(device)
        #self.emd_and_tree_wr = nn.Linear(self.model.config.hidden_size+self.args.tree_info_dim, 
                                          #self.model.config.hidden_size, bias=True)

        if checkpoint is not None:
            print ('Load parameters from checkpoint...')
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if abs_finetune is not None:
                print ('Load parameters from abs_finetune...')
                tree_params = [(n[8:], p) for n, p in abs_finetune['model'].items() if n.startswith('encoder')]
                self.encoder.load_state_dict(dict(tree_params), strict=True)
                tree_params = [(n[8:], p) for n, p in abs_finetune['model'].items() if n.startswith('decoder')]
                self.decoder.load_state_dict(dict(tree_params), strict=True)
                tree_params = [(n[10:], p) for n, p in abs_finetune['model'].items() if n.startswith('generator')]
                self.generator.load_state_dict(dict(tree_params), strict=True)
            else:
                print ('Initialize parameters for generator...')
                for p in self.generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                    else:
                        p.data.zero_()
                print ('Initialize parameters for TreeInference...')
                if args.param_init != 0.0 and self.planning_layer is not None:
                    for p in self.planning_layer.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                if args.param_init_glorot and self.planning_layer is not None:
                    for p in self.planning_layer.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)
                print ('Initialize parameters for TreeInference...')
                if args.param_init != 0.0:
                    for p in self.tree_info_wr.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                    for p in self.maxpool_linear.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                    for p in self.predicate_linear.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                    #for p in self.emd_and_tree_wr.parameters():
                    #    p.data.uniform_(-args.param_init, args.param_init)
                if args.param_init_glorot:
                    for p in self.tree_info_wr.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)
                    for p in self.maxpool_linear.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)
                    for p in self.predicate_linear.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)
                    #for p in self.emd_and_tree_wr.parameters():
                    #    if p.dim() > 1:
                    #        xavier_uniform_(p[:, :self.model.config.hidden_size])
                    #        zeros_(p[:, self.model.config.hidden_size:])
        self.to(device)


    def forward(self, src, tgt, mask_src, mask_tgt, 
                mask_src_sent=None, mask_tgt_sent=None, 
                mask_src_predicate=None, tgt_nsent=None, 
                labels=None, gt_aj_matrix=None, run_decoder=True):

        # Run encoder
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state
        content_selection_weights = mask_src
       
        # Extract embedding for sentence(or triples)
        if self.args.sentence_embedding == 'predicate':
            sents_vec = _get_predicate_embedding(top_vec, mask_src_predicate, mask_cls)
        if self.args.sentence_embedding == 'predicate_wordemb':
            inputs_embeds = self.encoder.embed_tokens(src)
            sents_vec = _get_predicate_embedding(inputs_embeds, mask_src_predicate, mask_cls, self.predicate_linear)
        elif self.args.sentence_embedding == 'predicate_maxpool':
            sents_vec = _get_sentence_maxpool(top_vec, mask_src_predicate)
            sents_vec = self.maxpool_linear(sents_vec)
        elif self.args.sentence_embedding == 'predicate_meanpool':
            sents_vec = _get_sentence_meanpool(top_vec, mask_src_predicate)
        elif self.args.sentence_embedding == 'meanpool':
            sents_vec = _get_sentence_meanpool(top_vec, mask_src_sent)
        else:
            sents_vec = _get_sentence_maxpool(top_vec, mask_src_sent)
            sents_vec = self.maxpool_linear(sents_vec)
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        # Get adjacency matrix
        if self.args.planning_method == 'ground_truth':
            aj_matrixes = gt_aj_matrix
            roots = None
        elif self.args.planning_method == 'self_attn':
            aj_matrixes = self.planning_layer(sents_vec, sents_vec, mask=mask_cls)
            roots = None
        else:
            roots, aj_matrixes = self.planning_layer(sents_vec, mask_cls)
            aj_matrixes = aj_matrixes[-1]
            # Softmax roots
            roots = roots[-1]
            reverse_mask = (~ mask_cls).bool()
            roots = roots.masked_fill(reverse_mask, float('-inf'))
            roots = self.softmax(roots)

        # Softmax aj_matrixes
        if self.args.planning_method != 'ground_truth':
           mask_cls_attn = (~ mask_cls).unsqueeze(1).expand_as(aj_matrixes).bool()
           aj_matrixes = aj_matrixes.masked_fill(mask_cls_attn, float('-inf'))
           aj_matrixes = self.softmax(aj_matrixes)
        raw_aj_matrixes = aj_matrixes

        # Edge sampling
        if self.args.gumbel_tau > 0:
            tau = self.args.gumbel_tau
            aj_matrixes = gumbel_softmax_function(aj_matrixes.transpose(1,2), tau, 1).transpose(1,2)

        # Get children/parents token embedding
        root_and_end_embeddings = self.encoder.embed_tokens(self.root_and_end_ids)
        root_embedding = root_and_end_embeddings[0]
        child_end_embedding = root_and_end_embeddings[1]
        # Get children weighted embedding
        children_embs = torch.matmul(aj_matrixes, sents_vec)
        child_end_embedding = child_end_embedding.unsqueeze(0).unsqueeze(0).expand_as(children_embs)
        children_embs += child_end_embedding * 1e-8
        # Get parents weighted embedding
        parents_embs = torch.matmul(aj_matrixes.transpose(1, 2), sents_vec)
        root_embedding = root_embedding.unsqueeze(0).expand(parents_embs.size(0), parents_embs.size(-1))
        root_embedding = root_embedding.unsqueeze(1)
        if roots is not None:
            roots = roots.unsqueeze(2)
            parents_embs = parents_embs + torch.matmul(roots, root_embedding)
        else:
            parents_embs += root_embedding * 1e-8
        
        # Merge tree information
        tree_info_embs = torch.cat([sents_vec, children_embs, parents_embs], dim=-1)
        tree_info_embs = self.tree_info_wr(tree_info_embs)
        #tree_info_embs = self.tree_info_layer_norm(tree_info_embs)
        tree_info_embs = self.tree_info_tanh(tree_info_embs)

        # Add embedding of tree information to the token embedding
        top_vec = top_vec.unsqueeze(1).repeat((1, mask_src_sent.size(1), 1, 1))
        tree_info_embs = tree_info_embs.unsqueeze(2).repeat((1, 1, src.size(1), 1))
        top_vec = top_vec + tree_info_embs
        #top_vec = self.emd_and_tree_wr(torch.cat([top_vec, tree_info_embs], dim=-1))
        top_vec = top_vec * mask_src_sent[:, :, :, None].float()
        top_vec = top_vec.sum(dim=1)

        if not run_decoder:
            return {"encoder_outpus":top_vec, 
                    "encoder_attention_mask":content_selection_weights,
                    "sent_probs":roots,
                    "sent_relations":raw_aj_matrixes}

        # Decoding
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=content_selection_weights)

        return decoder_outputs.last_hidden_state
