#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import json
import math
import random
import torch
from models.beam_search.beam import GNMTGlobalScorer
from transformers import (BeamSearchScorer, 
                          LogitsProcessorList, 
                          StoppingCriteriaList)

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
    translator = Translator(args, model, tokenizer, logger=logger)
    return translator


class Translator(object):

    def __init__(self, args, model, tokenizer, logger=None, dump_beam=""):

        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.start_token_id = self.tokenizer.bos_token_id
        self.end_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.end_token = self.tokenizer.eos_token
        if self.tokenizer.cls_token_id is None:
            self.cls_token = self.tokenizer.eos_token
        else:
            self.cls_token = self.tokenizer.cls_token

        self.beam_size = args.beam_size
        self.min_length = args.test_min_length
        self.max_length = args.test_max_length
        self.dump_beam = dump_beam
        self.device = self.model.device


    def translate(self, data_iter, step, attn_debug=False):
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        cluster_path = self.args.result_path + '.%d.cluster' % step
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        src_cluster_path = self.args.result_path + '.%d.src_cluster' % step
        eid_path = self.args.result_path + '.%d.eid' % step

        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.cluster_out_file = codecs.open(cluster_path, 'w', 'utf-8')
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        self.src_cluster_out_file = codecs.open(src_cluster_path, 'w', 'utf-8')
        self.eid_out_file = codecs.open(eid_path, 'w', 'utf-8')

        self.model.eval()
        with torch.no_grad():
            for batch in data_iter:
                # pridiction
                batch_data = self.translate_batch(batch)
                # prepare output data
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred_str, gold_str, pred_group, gold_group, group_score, src_group, src_str, eid = trans
                    self.gold_out_file.write(gold_str.strip() + '\n')
                    self.can_out_file.write(pred_str.strip() + '\n')
                    self.cluster_out_file.write(eid+'\t'+str(group_score)+'\t'+pred_group+'\t'+gold_group+'\n')
                    self.src_out_file.write(src_str.strip() + '\n')
                    self.src_cluster_out_file.write(src_group.strip() + '\n')
                    self.eid_out_file.write(eid + '\n')

                self.gold_out_file.flush()
                self.can_out_file.flush()
                self.cluster_out_file.flush()
                self.src_out_file.flush()
                self.src_cluster_out_file.flush()
                self.eid_out_file.flush()

        self.gold_out_file.close()
        self.can_out_file.close()
        self.cluster_out_file.close()
        self.src_out_file.close()
        self.src_cluster_out_file.close()
        self.eid_out_file.close()


    def from_batch(self, translation_batch):

        batch = translation_batch["batch"]
        batch_size = batch.batch_size

        preds = translation_batch["predictions"]
        pred_score = translation_batch["scores"]
        pred_clusters = translation_batch["pred_clusters"]
        pred_str_clusters = translation_batch["pred_str_clusters"]
        src_clusters = translation_batch["src_clusters"]
        cluster_probs = translation_batch["cluster_probs"]
        ngroups = translation_batch["ngroups"]
    
        translations = []
        for b in range(batch_size):

            # predicted sentence
            token_ids = preds[b][0]
            pred_sent = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            pred_sent = pred_sent.replace(self.end_token, '<q>')

            # predicted src groups
            src_cluster = src_clusters[b]
            src_cluster = ' ||| '.join([self.tokenizer.decode(s, skip_special_tokens=True) for s in src_cluster])

            # predicted predicate groups
            #pred_group = ' ||| '.join([' '.join(self.tokenizer.convert_ids_to_tokens(group)) for group in pred_clusters[b]])
            #gold_group = ' ||| '.join([' '.join(self.tokenizer.convert_ids_to_tokens(group)) for group in batch.p2s[b]])
            pred_group = ' ||| '.join([' '.join(item) for item in pred_str_clusters[b]])
            gold_group = ' ||| '.join([' '.join(item) for item in batch.pred_group_str[b]])

            gold_sent = '<q>'.join(batch.tgt_str[b])
            raw_src =  '\t'.join(batch.src_str[b])

            translation = (pred_sent, 
                           gold_sent, 
                           pred_group,
                           gold_group,
                           cluster_probs[b],
                           src_cluster,
                           raw_src, 
                           batch.eid[b])

            translations.append(translation)

        return translations


    def translate_batch(self, batch, fast=False):
        with torch.no_grad():
            return self._fast_translate_batch(batch)

    
    def _order_groups(self, input_args):
        src_groups, pred_groups, graph_probs, src_str_groups, pred_str_groups, _ = input_args
        '''
        print (src_groups)
        print (pred_groups)
        print (graph_probs)
        print (src_str_groups)
        print (pred_str_groups)
        print ('')
        '''

        main_subs = []
        for group in src_str_groups:
            if len(group) == 0:
                main_subs.append('')
                continue
            sub_freq = {}; obj_freq = {}
            for triple in group:
                sub_idx = triple.find('<SUB>')
                obj_idx = triple.find('<OBJ>')
                pred_idx = triple.find('<PRED>')
                sub = triple[sub_idx+5:pred_idx].strip()
                obj = triple[obj_idx+5:].strip()
                if sub not in sub_freq:
                    sub_freq[sub] = 0
                sub_freq[sub] += 1
                if obj not in obj_freq:
                    obj_freq[obj] = 0
                obj_freq[obj] += 1
            for key in sub_freq:
                if key not in obj_freq:
                    continue
                sub_freq[key] -= obj_freq[key]
            main_sub = sorted(sub_freq.items(), key = lambda d:d[1], reverse=True)[0][0]
            main_subs.append(main_sub)

        deterministic_model = self.model.deterministic_graph.model
        first_sent_lable = self.model.deterministic_graph.FIRST_SENT_LABEL
        first_sent_freq = {}
        for i, group in enumerate(pred_str_groups):
            if len(group) == 0:
                continue
            freq_sum = 0
            for pred in group:
                if pred in deterministic_model[first_sent_lable]:
                    first_freq = deterministic_model[first_sent_lable][pred]
                    freq_sum += first_freq
                else:
                    freq_sum += 0
            idx = len(first_sent_freq)
            first_sent_freq[idx] = freq_sum / float(len(group))

        new_best_candidate_same_sub = []; new_best_candidate_not_same_sub = []
        main_sub = ''
        for item in sorted(first_sent_freq.items(), key = lambda d:d[1], reverse=True):
            sentence_idx = item[0]
            current_group = pred_str_groups[sentence_idx]
            if main_sub == '':
                main_sub = main_subs[sentence_idx]
                new_best_candidate_same_sub.append(sentence_idx)
            else:
                max_sub = main_subs[sentence_idx]
                if max_sub == main_sub:
                    new_best_candidate_same_sub.append(sentence_idx)
                else:
                    new_best_candidate_not_same_sub.append(sentence_idx)
        sorted_sent = new_best_candidate_same_sub + new_best_candidate_not_same_sub

        src_groups = [src_groups[sentence_idx] for sentence_idx in sorted_sent]
        pred_groups = [pred_groups[sentence_idx] for sentence_idx in sorted_sent]
        graph_probs = [graph_probs[sentence_idx] for sentence_idx in sorted_sent]
        pred_str_groups = [pred_str_groups[sentence_idx] for sentence_idx in sorted_sent]

        return src_groups, pred_groups, graph_probs, pred_str_groups


    def _cluster_and_select(self, s, p, n_clusters, p_s, p_tok, p_tok_m, s_str, p_str):

        if self.args.test_alignment_type == 'random_test':
            n_clusters = random.randint(1, len(s))
            res = self.model.run_clustering(s, p, n_clusters, p_s, 
                                            p_tok, p_tok_m,
                                            mode=self.args.test_alignment_type, 
                                            src_str=s_str, pred_str=p_str)
            src_groups, pred_groups, graph_probs, pred_str_groups = self._order_groups(res)
            return src_groups, pred_groups, graph_probs, pred_str_groups

        if self.args.test_given_nclusters:
            res = self.model.run_clustering(s, p, n_clusters, p_s, 
                                            p_tok, p_tok_m,
                                            mode=self.args.test_alignment_type, 
                                            src_str=s_str, pred_str=p_str,
                                            run_bernoulli=self.args.test_run_bernoulli,
                                            adja_threshold=self.args.test_adja_threshold)
            src_groups, pred_groups, graph_probs, pred_str_groups = self._order_groups(res)
            return src_groups, pred_groups, graph_probs, pred_str_groups

        for i in range(1, len(s)+1):
            n_clusters = i
            res = self.model.run_clustering(s, p, n_clusters, p_s, 
                                            p_tok, p_tok_m,
                                            mode=self.args.test_alignment_type, 
                                            src_str=s_str, pred_str=p_str,
                                            run_bernoulli=self.args.test_run_bernoulli,
                                            adja_threshold=self.args.test_adja_threshold)
            src_groups, pred_groups, graph_probs, pred_str_groups = self._order_groups(res)

            if self.args.test_alignment_type != 'discriministic':
                graph_probs = [torch.exp(score) for score in graph_probs]

            graph_score = min(graph_probs)
            #if graph_score * len(s) >= self.args.test_graph_selection_threshold:
            if graph_score >= self.args.test_graph_selection_threshold:
            #if graph_score > self.args.test_graph_selection_threshold:
                return src_groups, pred_groups, graph_probs, pred_str_groups

        return src_groups, pred_groups, graph_probs, pred_str_groups


    def _run_clustering(self, src, preds, p2s, pred_tokens, pred_mask_tokens, src_str, pred_str, eids):

        # run clustering
        parallel_src = []
        parallel_src_example = []
        parallel_pred = []
        parallel_pred_str = []
        parallel_graph_probs = []
        ngroups = []

        for i in range(len(preds)):
            s = src[i] # src sentences
            s_str = src_str[i]
            p = preds[i]
            p_s = p2s[i]
            p_tok = pred_tokens[i]
            p_tok_m = pred_mask_tokens[i]
            p_str = pred_str[i].split(' ')
            n_clusters = len(p_s)

            if self.args.test_alignment_type == 'full_src':
                n_clusters = 1

            #print ('\n')
            #print (eids[i])
            ret = self._cluster_and_select(s, p, n_clusters, p_s, p_tok, p_tok_m, s_str, p_str)
            src_groups, pred_groups, graph_probs, pred_str_groups = ret

            parallel_src_example.append(src_groups)
            parallel_src.extend(src_groups)
            parallel_graph_probs.append(graph_probs)
            parallel_pred.append(pred_groups)
            parallel_pred_str.append(pred_str_groups)
            #ngroups.append(len(p_s))
            ngroups.append(len(pred_groups))

        src = torch.tensor(self.model._pad(parallel_src), device=self.device)
        mask_src = ~(src == self.pad_token_id)

        return src, mask_src, parallel_src_example, parallel_pred, parallel_pred_str, parallel_graph_probs, ngroups


    def _run_encoder(self, src, mask_src, ngroups):

        encoder = self.model.abs_model.get_encoder()
        encoder_outputs = encoder(input_ids=src, attention_mask=mask_src)
        src_features = encoder_outputs.last_hidden_state

        src_features_for_each_example = []
        mask_src_for_each_example = []
        src_for_each_example = []
        sid = 0
        for example_group_number in ngroups:
            eid = sid + example_group_number
            src_features_for_each_example.append(src_features[sid:eid])
            mask_src_for_each_example.append(mask_src[sid:eid])
            src_for_each_example.append(src[sid:eid])
            sid = sid + example_group_number

        return src_features_for_each_example, mask_src_for_each_example, src_for_each_example


    def _run_conditioned_text_generation(self, src_features, mask_src_features):

        model = self.model.abs_model
        beam_size = self.beam_size

        input_ids = torch.full([1, 1], self.start_token_id, device=self.device, dtype=torch.long)

        scores = []
        for i in range(src_features.size(0)):
            input_ids = input_ids.repeat_interleave(beam_size, dim=0)

            s_features = src_features[i]
            s_features = s_features.unsqueeze(0)
            s_features = s_features.repeat_interleave(beam_size, dim=0)

            mask_s_features = mask_src_features[i]
            mask_s_features = mask_s_features.unsqueeze(0)
            mask_s_features = mask_s_features.repeat_interleave(beam_size, dim=0)

            beam_scorer = BeamSearchScorer(batch_size=1, 
                                       num_beams=self.beam_size, 
                                       device=self.device)

            stopping_criteria = model._get_stopping_criteria(max_length=self.max_length, 
                                                         max_time=None, 
                                                         stopping_criteria=StoppingCriteriaList())

            logits_processor = model._get_logits_processor(repetition_penalty=model.config.repetition_penalty,
                                            no_repeat_ngram_size=model.config.no_repeat_ngram_size,
                                            encoder_no_repeat_ngram_size=model.config.encoder_no_repeat_ngram_size,
                                            input_ids_seq_length=input_ids.shape[-1],
                                            encoder_input_ids=input_ids,
                                            bad_words_ids=model.config.bad_words_ids,
                                            min_length=self.min_length,
                                            max_length=self.max_length,
                                            eos_token_id=self.end_token_id,
                                            forced_bos_token_id=model.config.forced_bos_token_id,
                                            forced_eos_token_id=model.config.forced_eos_token_id,
                                            prefix_allowed_tokens_fn=None,
                                            num_beams=beam_size,
                                            num_beam_groups=None,
                                            diversity_penalty=None,
                                            remove_invalid_values=model.config.remove_invalid_values,
                                            exponential_decay_length_penalty=model.config.exponential_decay_length_penalty,
                                            logits_processor=LogitsProcessorList())

            results = model.beam_search(input_ids,
                          beam_scorer,
                          logits_processor=logits_processor,
                          stopping_criteria=stopping_criteria,
                          pad_token_id=self.pad_token_id,
                          eos_token_id=self.end_token_id,
                          output_scores=True,
                          return_dict_in_generate=True,
                          synced_gpus=False,
                          encoder_outputs=[s_features],
                          attention_mask=mask_s_features)

            # /home/hpcxu1/miniconda3/envs/Plan/lib/python3.6/site-packages/transformers//generation_utils.py

            input_ids = results['sequences']
            scores.append(results['sequences_scores'])

        return input_ids[0], sum(scores)


    def _fast_translate_batch(self, batch):

        src = batch.src
        tgt = batch.tgt
        mask_tgt = batch.mask_tgt
        ctgt = batch.ctgt
        mask_ctgt = batch.mask_ctgt
        preds = batch.pred
        pred_tokens = batch.pred_tokens
        pred_mask_tokens = batch.pred_mask_tokens
        p2s = batch.p2s
        eids = batch.eid
        src_str = batch.src_str
        pred_str = batch.pred_str

        # run clustering
        ret = self._run_clustering(src, preds, p2s, pred_tokens, pred_mask_tokens, src_str, pred_str, eids)
        src, mask_src, src_examples, pred_clusters, pred_str_clusters, cluster_probs, ngroups = ret

        # run encoding
        ret = self._run_encoder(src, mask_src, ngroups)
        src_features_for_each_example, mask_src_for_each_example, src_for_each_example = ret
        
        # run decoding
        batch_size = batch.batch_size
        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["batch"] = batch
        results["pred_clusters"] = pred_clusters
        results["pred_str_clusters"] = pred_str_clusters
        results["src_clusters"] = src_examples
        results["cluster_probs"] = [sum(item) for item in cluster_probs]
        results["ngroups"] = ngroups

        # temp solution: run decoder one by one
        for batch_id in range(batch_size):
            src_features = src_features_for_each_example[batch_id]
            mask_src_features = mask_src_for_each_example[batch_id]
            prediction, score = self._run_conditioned_text_generation(src_features, mask_src_features)
            results["predictions"][batch_id].append(prediction)
            results["scores"][batch_id].append(score)

        return results

        '''
        batch_size = batch.batch_size
        results = {}

        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["batch"] = batch

        outputs = self.model.abs_model.generate(input_ids=src,
                                                    attention_mask=mask_src,
                                                    do_sample=False,
                                                    max_length=256,
                                                    min_length=5,
                                                    num_beams=5,
                                                    no_repeat_ngram_size=5)
        for i in range(outputs.size(0)):
            results["predictions"][i].append(outputs[i])
            results["scores"][i].append(0.0)

        return results

        '''

