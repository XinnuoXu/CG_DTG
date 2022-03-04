import copy

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from models.encoder import Classifier, TreeInference, SentenceClassification
from models.optimizers import Optimizer
from transformers import AutoModelForSeq2SeqLM

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
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
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))
    return optim


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.encoder = self.model.get_encoder()
        if args.content_planning_model == 'tree':
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
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.planning_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.planning_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        self.to(device)

    def forward(self, src, tgt, mask_src, mask_tgt, clss, mask_cls, gt_selection):
        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        # return transformers.modeling_outputs.BaseModelOutput
        top_vec = encoder_outputs.last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]

        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.planning_layer(sents_vec, mask_cls)
        return sent_scores, mask_cls


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)
    return generator


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, cls_token_id, checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        self.vocab_size = self.model.config.vocab_size
        self.cls_token_id = cls_token_id
        self.tree_gumbel_softmax_tau = args.tree_gumbel_softmax_tau

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
        else:
            self.planning_layer = None 

        self.decoder = self.model.get_decoder()
        self.generator = get_generator(self.vocab_size, self.model.config.hidden_size, device)
        self.generator[0].weight = self.decoder.embed_tokens.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
        self.to(device)


    def from_tree_to_mask(self, roots, input_ids, attention_mask, mask_block):
        root_probs = torch.sum(torch.stack(roots), 0)/len(roots)
        root_probs = root_probs * mask_block
        root_probs = nn.functional.gumbel_softmax(root_probs * mask_block, tau=self.tree_gumbel_softmax_tau, hard=False)
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
                #content_selection.append(weights[j].expand(sep_indices[j+1]-sep_indices[j]))
                content_selection.append(weights[j].repeat(sep_indices[j+1]-sep_indices[j]))
            content_selection_weights.append(torch.cat(content_selection))
        content_selection_weights = torch.stack(content_selection_weights)
        content_selection_weights = content_selection_weights * attention_mask
        return content_selection_weights


    def forward(self, src, tgt, mask_src, mask_tgt, clss, mask_cls, gt_selection):

        encoder_outputs = self.encoder(input_ids=src, attention_mask=mask_src) 
        top_vec = encoder_outputs.last_hidden_state

        if self.planning_layer != None:
            # Get sentence importance
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            sents_vec = sents_vec * mask_cls[:, :, None].float()
            sent_scores = self.planning_layer(sents_vec, mask_cls)
            # Weight input tokens
            content_selection_weights = self.from_tree_to_mask(sent_scores, src, mask_src, mask_cls)
        else:
            content_selection_weights = mask_src

        # Decoding
        decoder_outputs = self.decoder(input_ids=tgt, 
                                       attention_mask=mask_tgt,
                                       encoder_hidden_states=top_vec,
                                       encoder_attention_mask=content_selection_weights)

        return decoder_outputs.last_hidden_state
