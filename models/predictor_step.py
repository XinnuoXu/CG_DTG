#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import json
import math
import torch
from models.beam_search.beam import GNMTGlobalScorer
from models.tree_reader import tree_building, headlist_to_string, tree_to_mask_list
from tool.analysis_edge import Analysis

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def build_predictor(args, tokenizer, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')
    translator = Translator(args, model, tokenizer, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):

    def __init__(self, args, model, tokenizer, global_scorer=None, logger=None, dump_beam=""):

        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.generator = self.model.generator
        self.start_token_id = self.tokenizer.bos_token_id
        self.end_token_id = self.tokenizer.eos_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.cls_token = self.tokenizer.cls_token

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.test_min_length
        self.max_length = args.test_max_length
        self.dump_beam = dump_beam

        self.model_analysis = Analysis()


    def translate(self, data_iter, step, attn_debug=False):
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        eid_path = self.args.result_path + '.%d.eid' % step
        alg_path = self.args.result_path + '.%d.alignments' % step

        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        self.eid_out_file = codecs.open(eid_path, 'w', 'utf-8')
        self.alg_out_file = codecs.open(alg_path, 'w', 'utf-8')

        self.model.eval()
        with torch.no_grad():
            for batch in data_iter:
                # pridiction
                batch_data = self.translate_batch(batch)
                # prepare output data
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred_str, gold_str, src_str, src_list, eid, alignments, prompt_str = trans
                    self.can_out_file.write(pred_str.strip() + '\n')
                    self.gold_out_file.write(gold_str.strip() + '\n')
                    self.src_out_file.write(src_str.strip() + '\n')
                    self.eid_out_file.write(eid + '\n')
                    self.alg_out_file.write(prompt_str + '\n')

                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.eid_out_file.flush()
                self.alg_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.eid_out_file.close()
        self.alg_out_file.close()


    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["scores"]) == len(translation_batch["predictions"]))
        batch_size = batch.batch_size
        preds = translation_batch["predictions"]
        pred_score = translation_batch["scores"]
    
        src_str = batch.src_str
        tgt_str = batch.tgt_str
        src = batch.src
        eid = batch.eid
        alg = batch.alg
        prompt_str = batch.prompt_str

        translations = []
        for b in range(batch_size):
            token_ids = preds[b][0]
            pred_sent = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            pred_sent = pred_sent.replace(self.cls_token, '<q>')
            gold_sent = '<q>'.join(tgt_str[b])
            src_list = src_str[b]
            raw_src = self.tokenizer.decode(src[b], skip_special_tokens=False)

            alignments = []
            for sent in alg[b]:
                alignments.append(' ; '.join([src_list[idx] for idx in sent]))
            alignments = ' ||| '.join(alignments)

            translation = (pred_sent, 
                           gold_sent, 
                           raw_src, 
                           src_list, 
                           eid[b], 
                           alignments,
                           prompt_str[b])

            translations.append(translation)

        return translations


    def translate_batch(self, batch, fast=False):
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)


    def _fast_translate_batch(self, batch, max_length, min_length=0):

        assert not self.dump_beam
        beam_size = self.beam_size
        batch_size = batch.batch_size

        src = batch.src
        mask_src = batch.mask_src
        mask_src_sent = batch.mask_src_sent
        tgt = batch.tgt
        mask_tgt = batch.mask_tgt
        mask_tgt_sent = batch.mask_tgt_sent
        clss = batch.clss
        mask_cls = batch.mask_cls
        labels = batch.alg
        gt_aj_matrix = batch.gt_aj_matrix
        device = src.device
        results = {}

        # Run encoder and tree prediction
        src_res = self.model(src, tgt, mask_src, mask_tgt,
                                 mask_src_sent, mask_tgt_sent,
                                 clss, mask_cls, labels,
                                 run_decoder=False)
        src_features = src_res['encoder_outpus']
        mask_src = src_res['encoder_attention_mask']

        # Tile states and memory beam_size times.
        sentence_plans = tree_to_mask_list(labels, mask_src_sent)
        current_tgt_example_id = []
        for j in range(src.size(0)):
            current_tgt_example_id.extend([j] * beam_size)
        current_tgt_sentence = [0] * src.size(0)*beam_size
        current_tgt_example_id = torch.tensor(current_tgt_example_id, dtype=int, device=device)
        current_tgt_sentence = torch.tensor(current_tgt_sentence, dtype=int, device=device)
        content_weights = None
        current_sent_length = torch.zeros(batch_size * beam_size, device=device)

        src_features = tile(src_features, beam_size, dim=0)
        mask_src = tile(mask_src, beam_size, dim=0)

        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
        alive_seq = torch.full([batch_size * beam_size, 1], self.start_token_id, dtype=torch.long, device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["batch"] = batch

        for step in range(max_length):

            decoder_input = alive_seq

            content_weight = []
            for j, ex_id in enumerate(current_tgt_example_id):
                current_sent = current_tgt_sentence[j]
                # For some cases where the top generation in the beam is unfinished but some of the rest have finished
                if current_sent >= len(sentence_plans[ex_id]):
                    current_sent = -1
                content_weight.append(sentence_plans[ex_id][current_sent].unsqueeze(0))
            content_weight = torch.cat(content_weight, dim=0).unsqueeze(1)
            if content_weights is None:
                content_weights = content_weight
            else:
                content_weights = torch.cat([content_weights, content_weight], dim=1)

            decoder_outputs = self.model.decoder(input_ids=decoder_input,
                                           encoder_hidden_states=src_features,
                                           encoder_attention_mask=mask_src,
                                           content_weights=content_weights)

            dec_out = decoder_outputs.last_hidden_state[:, -1, :]

            # Generator forward.
            log_probs = self.generator.forward(dec_out)
            vocab_size = log_probs.size(-1)

            current_sent_length += 1
            for i in range(current_sent_length.size(0)):
                if current_sent_length[i] < min_length:
                    log_probs[i, self.cls_token_id] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            #alpha = self.global_scorer.alpha
            #length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            length_penalty = 1.0

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = self.tokenizer.decode(words).split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size).int()
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)
            content_weights = content_weights.index_select(0, select_indices)
            current_tgt_example_id = current_tgt_example_id.index_select(0, select_indices)
            current_tgt_sentence = current_tgt_sentence.index_select(0, select_indices)
            current_sent_length = current_sent_length.index_select(0, select_indices)
            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            # If one sentence finishes
            topk_ids = topk_ids.view(-1)
            sent_finished = topk_ids.eq(self.cls_token_id)
            for i in range(sent_finished.size(0)):
                if sent_finished[i]:
                    current_tgt_sentence[i] += 1
                    current_sent_length[i] = 0
                example_id = current_tgt_example_id[i]
                if current_tgt_sentence[i] >= len(sentence_plans[example_id]):
                    topk_ids[i] = self.end_token_id
            topk_ids = topk_ids.view(-1, beam_size)

            is_finished = topk_ids.eq(self.end_token_id)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            mask_src = mask_src.index_select(0, select_indices)
            if is_finished.any():
                current_tgt_example_id = current_tgt_example_id.index_select(0, select_indices)
                current_tgt_sentence = current_tgt_sentence.index_select(0, select_indices)
                content_weights = content_weights.index_select(0, select_indices)
                current_sent_length = current_sent_length.index_select(0, select_indices)

        return results


