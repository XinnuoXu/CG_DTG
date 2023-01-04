#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import json
import math
import torch
from models.beam_search.beam import GNMTGlobalScorer

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
        self.start_token_id = self.tokenizer.bos_token_id
        self.end_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.end_token = self.tokenizer.eos_token

        self.global_scorer = global_scorer
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
            pred_group = ' ||| '.join([' '.join(self.tokenizer.convert_ids_to_tokens(group)) for group in pred_clusters[b]])
            gold_group = ' ||| '.join([' '.join(self.tokenizer.convert_ids_to_tokens(group)) for group in batch.p2s[b]])

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
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    
    def _order_groups(self, input_args):
        src_groups, pred_groups, graph_probs, src_str_groups, pred_str_groups = input_args

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
            first_sent_freq[i] = freq_sum / float(len(group))

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

        return src_groups, pred_groups, graph_probs


    def _cluster_and_select(self, s, p, n_clusters, p_s, s_str, p_str):

        if self.args.test_given_nclusters:
            res = self.model.run_clustering(s, p, n_clusters, p_s, 
                                            mode=self.args.test_alignment_type, 
                                            src_str=s_str, pred_str=p_str)
            src_groups, pred_groups, graph_probs = self._order_groups(res)
            return src_groups, pred_groups, graph_probs

        for i in range(1, len(s)+1):
            n_clusters = i
            res = self.model.run_clustering(s, p, n_clusters, p_s, 
                                            mode=self.args.test_alignment_type, 
                                            src_str=s_str, pred_str=p_str)
            src_groups, pred_groups, graph_probs = self._order_groups(res)

            graph_score = min(graph_probs)
            if graph_score >= self.args.test_graph_selection_threshold:
                return src_groups, pred_groups, graph_probs

        return src_groups, pred_groups, graph_probs


    def _run_clustering(self, src, preds, p2s, src_str, pred_str):

        # run clustering
        parallel_src = []
        parallel_src_example = []
        parallel_pred = []
        parallel_graph_probs = []
        ngroups = []

        for i in range(len(preds)):
            s = src[i] # src sentences
            s_str = src_str[i]
            p = preds[i]
            p_s = p2s[i]
            p_str = pred_str[i].split(' ')
            n_clusters = len(p_s)

            src_groups, pred_groups, graph_probs = self._cluster_and_select(s, p, n_clusters, p_s, s_str, p_str)

            parallel_src_example.append(src_groups)
            parallel_src.extend(src_groups)
            parallel_graph_probs.append(graph_probs)
            parallel_pred.append(pred_groups)
            #ngroups.append(len(p_s))
            ngroups.append(len(pred_groups))

        src = torch.tensor(self.model._pad(parallel_src), device=self.device)
        mask_src = ~(src == self.pad_token_id)

        return src, mask_src, parallel_src_example, parallel_pred, parallel_graph_probs, ngroups


    def _run_encoder(self, src, mask_src, ngroups):

        encoder_outputs = self.model.abs_model.encoder(input_ids=src, attention_mask=mask_src)
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


    def _run_conditioned_text_generation(self, src, mask_src, src_examples,
                                         pred_clusters, cluster_probs, ngroups,
                                         src_features_for_each_example,
                                         mask_src_for_each_example,
                                         src_for_each_example,
                                         max_length, min_length, batch=None):

        device = self.device
        beam_size = self.beam_size
        batch_size = batch.batch_size

        # prepare for decoder
        current_group_ids = torch.zeros(beam_size * len(src_features_for_each_example), device=device)
        ngroup_for_each_example = torch.tensor(ngroups, device=device)
        ngroup_for_each_example = tile(ngroup_for_each_example, beam_size, dim=0)
        current_sent_length = torch.zeros(beam_size * batch_size, device=device)

        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
        alive_seq = torch.full([batch_size * beam_size, 1], self.start_token_id, dtype=torch.long, device=device)
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size))

        hypotheses = [[] for _ in range(batch_size)] # Structure that holds finished hypotheses.

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["batch"] = batch
        results["pred_clusters"] = pred_clusters
        results["src_clusters"] = src_examples
        #results["cluster_probs"] = [sum(item).item() for item in cluster_probs]
        results["cluster_probs"] = [sum(item) for item in cluster_probs]
        results["ngroups"] = ngroups

        for step in range(max_length):

            # reconstruct src_features
            src_features = []; mask_src = []; #tmp_src = []
            for i in range(current_group_ids.size(0)):
                example_id = int(i/beam_size)
                group_id = int(current_group_ids[i])
                src_features.append(src_features_for_each_example[example_id][group_id])
                #tmp_src.append(src_for_each_example[example_id][group_id])
                mask_src.append(mask_src_for_each_example[example_id][group_id])
            src_features = torch.stack(src_features, 0)
            mask_src = torch.stack(mask_src, 0)

            '''
            print (current_group_ids)
            print (src_features[:,1,:3])
            print (mask_src)
            print ('\n')

            for i in range(alive_seq.size(0)):
                print (' '.join(self.tokenizer.convert_ids_to_tokens(tmp_src[i])).replace('<pad> ', ''))
                #print (tmp_src[i])
                #print (mask_src[i])
                print ('***')
                print (' '.join(self.tokenizer.convert_ids_to_tokens(alive_seq[i])))
                print ('--------------------------')
            print (current_group_ids)
            print (current_sent_length)
            print ('\n')
            '''

            # run decoder
            decoder_input = alive_seq
            decoder_outputs = self.model.abs_model.decoder(input_ids=decoder_input,
                                                           encoder_hidden_states=src_features,
                                                           encoder_attention_mask=mask_src)
            dec_out = decoder_outputs.last_hidden_state[:, -1, :]

            # generator forward.
            log_probs = self.model.abs_model.generator.forward(dec_out)
            vocab_size = log_probs.size(-1)

            current_sent_length += 1
            for i in range(log_probs.size(0)):
                if current_sent_length[i] < min_length:
                    log_probs[i, self.end_token_id] = -1e20

            # multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
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

            # Append last prediction.
            alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

            # Judge whether the generation of one sentence finished
            current_group_ids = current_group_ids.index_select(0, select_indices)
            ngroup_for_each_example = ngroup_for_each_example.index_select(0, select_indices)
            current_sent_length = current_sent_length.index_select(0, select_indices)

            ## if finished one sentence, move on to the next sentence
            finish_one_sent = topk_ids.eq(self.end_token_id)
            current_group_ids = current_group_ids.reshape(-1, beam_size)
            current_group_ids += finish_one_sent

            ## if finished one sentence, set current sentence generation length to 0
            current_sent_length = current_sent_length.reshape(-1, beam_size)
            current_sent_length = current_sent_length * (~finish_one_sent)

            ## if all the input clusters are generated already, set the generation of the candidate done
            ngroup_for_each_example = ngroup_for_each_example.reshape(-1, beam_size)
            is_finished = (current_group_ids >= ngroup_for_each_example)
            topk_log_probs = is_finished * (-1e20) + topk_log_probs # Dangourse!!!
            current_group_ids = is_finished * (-1) + current_group_ids # Dangourse!!!

            ## reset the shapes
            current_group_ids = current_group_ids.view(-1)
            ngroup_for_each_example = ngroup_for_each_example.view(-1)
            current_sent_length = current_sent_length.view(-1)

            # If any examples finished
            if step + 1 == max_length:
                is_finished.fill_(1)
            end_condition = is_finished[:, 0].eq(1) # End condition is top beam is finished.

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
                if len(non_finished) == 0:
                    # If all sentences are translated, no need to go further.
                    break

                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
                src_features_for_each_example = [src_features_for_each_example[non_finished_id] for non_finished_id in non_finished]
                mask_src_for_each_example = [mask_src_for_each_example[non_finished_id] for non_finished_id in non_finished]

                current_group_ids = current_group_ids.view(-1, beam_size)
                current_group_ids = current_group_ids.index_select(0, non_finished)
                current_group_ids = current_group_ids.view(-1)

                ngroup_for_each_example = ngroup_for_each_example.view(-1, beam_size)
                ngroup_for_each_example = ngroup_for_each_example.index_select(0, non_finished)
                ngroup_for_each_example = ngroup_for_each_example.view(-1)

                current_sent_length = current_sent_length.view(-1, beam_size)
                current_sent_length = current_sent_length.index_select(0, non_finished)
                current_sent_length = current_sent_length.view(-1)

        return results


    def _fast_translate_batch(self, batch, max_length, min_length=0):

        src = batch.src
        tgt = batch.tgt
        mask_tgt = batch.mask_tgt
        ctgt = batch.ctgt
        mask_ctgt = batch.mask_ctgt
        preds = batch.pred
        p2s = batch.p2s

        src_str = batch.src_str
        pred_str = batch.pred_str

        # run clustering
        src, mask_src, src_examples, pred_clusters, cluster_probs, ngroups = self._run_clustering(src, preds, p2s, src_str, pred_str)

        # run encoding
        src_features_for_each_example, mask_src_for_each_example, src_for_each_example = self._run_encoder(src, mask_src, ngroups)
        
        # run decoding
        results = self._run_conditioned_text_generation(src, mask_src, src_examples,
                                                        pred_clusters, cluster_probs, ngroups,
                                                        src_features_for_each_example,
                                                        mask_src_for_each_example,
                                                        src_for_each_example,
                                                        max_length, min_length, batch=batch)

        return results
