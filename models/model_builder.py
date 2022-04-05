import copy, random

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from models.encoder import Classifier, TreeInference, SentenceClassification
from models.decoder import BartDecoderCS
from models.optimizers import Optimizer
from transformers import AutoModelForSeq2SeqLM

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
            #print (name)
            params.append((name, para))
    optim.set_parameters(params)
    return optim


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, vocab_size, checkpoint, content_planning_model):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.vocab_size = vocab_size
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)

        if self.vocab_size > self.model.config.vocab_size:
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        if content_planning_model == 'tree':
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


    def forward(self, src, tgt, mask_src, mask_tgt, clss, mask_cls, gt_selection):
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        # return transformers.modeling_outputs.BaseModelOutput
        top_vec = encoder_outputs.last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]

        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores, aj_matrixes = self.planning_layer(sents_vec, mask_cls)
        return sent_scores, mask_cls, aj_matrixes



def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)
    return generator



class AbsSummarizer(nn.Module):
    def __init__(self, args, device, cls_token_id, vocab_size, checkpoint=None, ext_checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = vocab_size
        self.cls_token_id = cls_token_id
        self.tree_gumbel_softmax_tau = args.tree_gumbel_softmax_tau

        if self.vocab_size > self.model.config.vocab_size:
            self.model.resize_token_embeddings(self.vocab_size)

        self.encoder = self.model.get_encoder()
        if args.content_planning_model == 'tree':
            self.planning_layer = TreeInference(self.model.config.hidden_size, 
                                                args.ext_ff_size, 
                                                args.ext_dropout, 
                                                args.ext_layers)
        elif args.content_planning_model == 'transformer':
            self.planning_layer = SentenceClassification(self.model.config.hidden_size, 
                                                         args.ext_ff_size, 
                                                         args.ext_heads, 
                                                         args.ext_dropout, 
                                                         args.ext_layers)
        elif args.content_planning_model == 'independent_tree_planner':
            self.planning_layer = ExtSummarizer()
        else:
            self.planning_layer = None 

        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, self.model.config.hidden_size, device)
        #self.generator[0].weight = self.model.lm_head.weight

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

            if self.planning_layer is not None:
                if (ext_checkpoint is not None):
                    print ('Load parameters from ext_checkpoint...')
                    tree_params = [(n[15:], p) for n, p in ext_checkpoint['model'].items() if n.startswith('planning_layer')]
                    self.planning_layer.load_state_dict(dict(tree_params), strict=True)
                else:
                    print ('Initialize parameters for ext_checkpoint...')
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


    def forward(self, src, tgt, mask_src, mask_tgt, clss, mask_cls, gt_selection, run_decoder=True):

        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state
        content_selection_weights = mask_src
        sent_scores_layers = None

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
        self.tree_gumbel_softmax_tau = args.tree_gumbel_softmax_tau

        # Encoder for Generator
        self.encoder = model.get_encoder()

        # Decoder for Generator
        self.decoder = model.get_decoder()
        self.generator = get_generator(self.vocab_size, model.config.hidden_size, device)

        # Planner (parameters are initialized)
        self.planning_layer = ExtSummarizer(args, device, ext_finetune, args.content_planning_model)

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


    def gumbel_softmax_function(self, scores, tau, top_k):
        top_k = int(top_k)
        gumbels = -torch.empty_like(scores.contiguous()).exponential_().log()
        gumbels = (scores + gumbels) / tau
        y_soft = gumbels.softmax(-1)
        top_k = min(top_k, y_soft.size(1))
        indices = torch.topk(y_soft, dim=-1, k=top_k)[1]
        value = 1 / top_k
        y_hard = torch.zeros_like(scores.contiguous()).scatter_(-1, indices, value)
        ret = y_hard - y_soft.detach() + y_soft
        ret = (ret == value)
        return ret


    def topn_function(self, scores, mask_block, top_n):
        if top_n >= 1:
            top_n = int(top_n)
            indices = torch.topk(scores, min(scores.size(1), top_n))[1]
            y_hard = torch.zeros_like(scores.contiguous()).scatter_(-1, indices, 1)
        else:
            y_hard = []
            nsent_selection = (mask_block.sum(dim=1) * top_n).int().tolist()
            for b in range(len(nsent_selection)):
                indices = torch.topk(scores[b], nsent_selection[b])[1]
                y_h = torch.zeros_like(scores[b].contiguous()).scatter_(-1, indices, 1)
                y_hard.append(y_h)
            y_hard = torch.stack(y_hard)
        return y_hard


    def from_tree_to_mask(self, roots, input_ids, attention_mask, mask_block, gt_selection):
        #root_probs = torch.sum(torch.stack(roots), 0)/len(roots)
        root_probs = roots[-1]
        root_probs = root_probs * mask_block
        if self.args.planning_method == 'ground_truth':
            root_probs = gt_selection
        elif self.args.planning_method == 'topk_tree':
            root_probs = self.topn_function(root_probs, mask_block, self.args.ext_topn)
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
            root_probs = self.gumbel_softmax_function(root_probs, self.tree_gumbel_softmax_tau, self.args.ext_topn)

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
                    "sent_relations":aj_matrixes}

        # Decoding for Generator
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=mask_src,
                                       content_selection_mask=content_selection_weights)

        return decoder_outputs.last_hidden_state, sent_scores_layers
