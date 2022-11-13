#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import json
import math
import torch
from models.beam_search.beam import GNMTGlobalScorer, BeamSearchDecoding, tile

def build_predictor(args, tokenizer, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')
    translator = Translator(args, model, tokenizer, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):

    def __init__(self, args, model, tokenizer, global_scorer=None, logger=None):

        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.generator = self.model.generator
        self.beam_search_generator = BeamSearchDecoding(self.model, self.generator, self.tokenizer, 
                                                        self.args.block_trigram, 
                                                        self.args.beam_size)

        self.start_token_id = self.tokenizer.bos_token_id
        self.end_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.plan_end_id = self.tokenizer.convert_tokens_to_ids(['|||'])[0]
        self.first_sentence_id = self.tokenizer.convert_tokens_to_ids(['<FIRST_SENT>'])[0]
        self.nonfirst_sentence_id = self.tokenizer.convert_tokens_to_ids(['<NOT_FIRST_SENT>'])[0]

        if self.tokenizer.cls_token_id is None:
            self.cls_token = self.tokenizer.eos_token
        else:
            self.cls_token = self.tokenizer.cls_token

        self.global_scorer = global_scorer
        self.min_length = args.test_min_length
        self.max_length = args.test_max_length


    def translate(self, data_iter, step, attn_debug=False):
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        eid_path = self.args.result_path + '.%d.eid' % step

        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        self.eid_out_file = codecs.open(eid_path, 'w', 'utf-8')

        self.model.eval()
        with torch.no_grad():
            for batch in data_iter:
                # pridiction
                batch_data = self._predict_batch(batch,
                                         max_length=self.max_length,
                                         min_length=self.min_length)
                # prepare output data
                translations = self._write_output(batch_data)

                for trans in translations:
                    pred_str, gold_str, src_str, src_list, eid = trans
                    self.can_out_file.write(pred_str.strip() + '\n')
                    self.gold_out_file.write(gold_str.strip() + '\n')
                    self.src_out_file.write(src_str.strip() + '\n')
                    self.eid_out_file.write(eid + '\n')

                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                self.eid_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()
        self.eid_out_file.close()


    def _write_output(self, translation_batch):

        assert (len(translation_batch["scores"]) == len(translation_batch["predictions"]))

        preds = translation_batch["predictions"]
        pred_score = translation_batch["scores"]
    
        src_str = translation_batch["src_txts"]
        tgt_str = translation_batch["tgt_txts"]
        src = translation_batch["src"]
        eid = translation_batch["eid"]

        batch_size = len(preds)

        translations = []
        for b in range(batch_size):
            token_ids = preds[b][0]

            pred_sent = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            pred_sent = pred_sent.replace(self.cls_token, '<q>')
            gold_sent = '<q>'.join(tgt_str[b])
            src_list = src_str[b]
            raw_src = self.tokenizer.decode(src[b], skip_special_tokens=False)

            translation = (pred_sent, 
                           gold_sent, 
                           raw_src, 
                           src_list, 
                           eid[b])

            translations.append(translation)

        return translations


    def _discrete_plan_classification(self, predicates, slot_nums=None):
        plans = []
        device = predicates[0].device
        for i, pred in enumerate(predicates):
            p_emb = self.model.encoder.embed_tokens(pred) # pred.size() = [number of predicates]
            p_emb = p_emb.unsqueeze(0) # (batch_size=1, length, dim)

            s_embs, attn = self.model.planner(p_emb, num_slots=slot_nums[i]) # (batch_size, slot_num, dim)

            slot_embs = s_embs.squeeze(dim=0).unsqueeze(dim=1) #(slot_num, 1, dim)
            for j in range(slot_embs.size(0)):
                slot_embs[j] = p_emb[0, j, :]
            slot_embs = [slot_embs[j].unsqueeze(1) for j in range(slot_embs.size(0))]

            attn = attn.squeeze(dim=0)
            attn_max = torch.max(attn, dim=0)[1]

            solutions = [[] for j in range(slot_nums[i])]
            for j in range(attn_max.size(0)):
                cluster_id = attn_max[j]
                pred_id = int(pred[j])
                solutions[cluster_id].append(pred_id)

            new_solutions = []; new_slot_embs = []
            for j, item in enumerate(solutions):
                if len(item) == 0:
                    continue
                if j == 0:
                    new_s = [self.start_token_id]+item + [self.first_sentence_id]
                else:
                    new_s = [self.start_token_id]+item + [self.nonfirst_sentence_id]
                new_solutions.append(new_s)
                new_slot_embs.append(slot_embs[j])
            solutions = [torch.tensor(item, dtype=torch.long, device=device).unsqueeze(0) for item in new_solutions]
            slot_embs = new_slot_embs

            res = (len(solutions), solutions, 0.0, slot_embs)
            plans.append(res)

        return plans


    def _marginal_plan_generation(self, predicates, slot_nums=None):
        plans = []
        device = predicates[0].device
        for i, pred in enumerate(predicates):
            p_emb = self.model.encoder.embed_tokens(pred) # pred.size() = [number of predicates]
            p_emb = p_emb.unsqueeze(0)
            s_embs, attn = self.model.planner(p_emb, num_slots=slot_nums[i]) # (batch_size, slot_num, dim)
            slot_embs = s_embs.squeeze(dim=0).unsqueeze(dim=1) #(slot_num, 1, dim)
            slot_embs = [slot_embs[j].unsqueeze(0) for j in range(slot_embs.size(0))]

            solutions = []
            for j in range(slot_nums[i]):
                if j == 0:
                    prompt = [self.start_token_id, self.first_sentence_id]
                else:
                    #prompt = [self.start_token_id, self.nonfirst_sentence_id]
                    prompt = [self.start_token_id, self.first_sentence_id]
                solutions.append(torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0))

            res = (len(solutions), solutions, 0.0, slot_embs)
            plans.append(res)

        return plans


    def _discrete_plan_generation(self, src_features, mask_src, predicates, slot_nums=None):
        device = src_features.device
        plans = []
        for i, pred in enumerate(predicates):
            p_emb = self.model.encoder.embed_tokens(pred) # pred.size() = [number of predicates]
            p_emb = p_emb.unsqueeze(0) # (batch_size=1, length, dim)

            candidate_nslot = []; candidate_solutions = []; candidate_scores = []; candidate_start_embs = []

            for nslot in range(1, pred.size(0)+1):
                if slot_nums is not None:
                    if nslot != slot_nums[i]:
                        continue
                s_embs, attn = self.model.planner(p_emb, num_slots=nslot) # (batch_size, slot_num, dim)
                slot_embs = s_embs.squeeze(dim=0).unsqueeze(dim=1) #(slot_num, 1, dim)

                top_vec = src_features[i].unsqueeze(dim=0)
                s_mask = mask_src[i].unsqueeze(dim=0)
                top_vec = tile(top_vec, nslot, dim=0) # (batch_size=slot_num, src_length, dim)
                s_mask = tile(s_mask, nslot, dim=0) # (batch_size=slot_num, src_length)

                # Run decoder
                alive_seq = torch.full([nslot, 1], self.start_token_id, dtype=torch.long, device=device) # (slot_num, 1)
                scores = torch.tensor([0.0] * nslot, device=device) # (slot_num)
                hypotheses = []
                for step in range(pred.size(0)+1):
                    tgt_embs = self.model.decoder.embed_tokens(alive_seq) # (slot_num, tgt_length, dim)
                    tgt_embs = torch.cat((slot_embs, tgt_embs[:,1:,:]), dim=1)
                    decoder_outputs = self.model.decoder(inputs_embeds=tgt_embs,
                                                         encoder_hidden_states=top_vec,
                                                         encoder_attention_mask=s_mask)
                    dec_out = decoder_outputs.last_hidden_state[:, -1, :] # (slot_num, dim)

                    log_probs = self.generator.forward(dec_out)
                    topk_scores, topk_ids = log_probs.topk(1, dim=-1) # topk_scores and topk_ids (slot_num, topk)
                    scores += topk_scores.squeeze(dim=1)

                    alive_seq = torch.cat([alive_seq, topk_ids], -1) #(slot_num, tgt_length)

                    is_finished = topk_ids.eq(self.plan_end_id)
                    if step + 1 == pred.size(0)+1:
                        alive_seq[:,-1] = self.plan_end_id
                        is_finished.fill_(1)

                    if is_finished.any():
                        is_finished = is_finished.reshape(-1)
                        finished_hyp = is_finished.nonzero()
                        for j in finished_hyp:
                            hypotheses.append((scores[j], alive_seq[j], slot_embs[j]))

                        non_finished = is_finished.eq(0).nonzero().reshape(-1)
                        if len(non_finished) == 0:
                            break
                        scores = scores.index_select(0, non_finished)
                        alive_seq = alive_seq.index_select(0, non_finished)
                        top_vec = top_vec.index_select(0, non_finished)
                        s_mask = s_mask.index_select(0, non_finished)
                        slot_embs = slot_embs.index_select(0, non_finished)

                candidate_nslot.append(nslot)
                candidate_scores.append(sum([pair[0] for pair in hypotheses]))
                candidate_solutions.append([pair[1] for pair in hypotheses])
                candidate_start_embs.append([pair[2] for pair in hypotheses])

            # store and sort for debugging
            zipped = zip(candidate_nslot, candidate_solutions, candidate_scores, candidate_start_embs)
            zipped = list(zipped)
            res = sorted(zipped, key = lambda x: x[2], reverse=True)
            bast_solution = res[0]
            plans.append(bast_solution)
        
        return plans


    def _generation_preprocess(self, plans, src, src_features, mask_src, pred, eids, src_txts, tgt_txts, device):
        # Expend src and tgt for subsentences
        new_src = []; new_src_features = []; new_mask_src = []
        new_tgt = []; new_prompts_control = []; new_init_embeddings = []
        new_src_txts = []; new_tgt_txts = []; new_example_ids = []

        max_plan_length = max([len(item) for item in pred]) + 2
        for i, example in enumerate(plans):
            nslot, plan, score, init_emb = example
            # src
            src_t = torch.cat([src[i].unsqueeze(0)] * nslot, dim=0) # slot_num * src_ntoken 
            src_emb = torch.cat([src_features[i].unsqueeze(0)] * nslot, dim=0) # slot_num * src_ntoken * dim
            src_m = torch.cat([mask_src[i].unsqueeze(0)] * nslot, dim=0) # slot_num * src_ntoken 
            # eid
            example_id = [eids[i]] * nslot
            example_id = [f'{id}_{k}' for k, id in enumerate(example_id)]
            # tgt
            init_emb = torch.cat(init_emb, dim=0) # slot_num * 1 * dim
            prompts = torch.full([nslot, max_plan_length], 
                                 self.pad_token_id, 
                                 dtype=torch.long, 
                                 device=device) # (slot_num, max_plan_length)
            for j, p in enumerate(plan):
                length = p.size(1)
                prompts[j][:length] = p
            prompts_control = (prompts == self.pad_token_id) # (slot_num, max_plan_length)

            # src_txts and tgt_txts
            new_src_txts.extend([src_txts[i]] * nslot)
            new_tgt_txts.extend([tgt_txts[i]] * nslot)

            new_src.append(src_t)
            new_src_features.append(src_emb)
            new_mask_src.append(src_m)
            new_example_ids.extend(example_id)
            new_tgt.append(prompts)
            new_prompts_control.append(prompts_control)
            new_init_embeddings.append(init_emb)

        src = torch.cat(new_src)
        src_features = torch.cat(new_src_features) # all_slot_num * src_ntoken * dim
        mask_src = torch.cat(new_mask_src) # all_slot_num * src_ntoken
        tgt = torch.cat(new_tgt) # all_slot_num * max_pred_num
        prompts_control = torch.cat(new_prompts_control) # all_slot_num * max_pred_num
        init_embeddings = torch.cat(new_init_embeddings) # all_slot_num * 1 * dim
        example_ids = new_example_ids
        src_txts = new_src_txts
        tgt_txts = new_tgt_txts

        return src, src_features, mask_src, tgt, prompts_control, init_embeddings, example_ids, src_txts, tgt_txts


    def _predict_batch(self, batch, max_length=128, min_length=0):

        src = batch.src
        mask_src = batch.mask_src
        tgt = batch.tgt
        mask_tgt = batch.mask_tgt
        mask_loss = batch.mask_loss
        pred = batch.pred
        nsent = batch.nsent
        eids = batch.eid
        src_txts = batch.src_str
        tgt_txts = batch.tgt_str
        device = src.device

        # Run encoder and tree prediction
        encoder_outputs = self.model.encoder(input_ids=src, attention_mask=mask_src)
        src_features = encoder_outputs.last_hidden_state

        # Predict plan
        plans = self._discrete_plan_classification(pred, slot_nums=nsent)
        #plans = self._marginal_plan_generation(pred, slot_nums=nsent)
        #plans = self._discrete_plan_generation(src_features, mask_src, pred, slot_nums=nsent)

        # Preprocess sentence generation
        preprocessed_data = self._generation_preprocess(plans, src, src_features, mask_src, pred, eids, src_txts, tgt_txts, device)
        src, src_features, mask_src, tgt, prompts_control, init_embeddings, example_ids, src_txts, tgt_txts = preprocessed_data

        # Generation text
        results = self.beam_search_generator.run(src, src_features, mask_src, 
                                                 tgt, prompts_control, 
                                                 src_txts, tgt_txts,
                                                 min_length, max_length, device, 
                                                 example_ids, init_embeddings=init_embeddings)

        return results
