import os
import json
import numpy as np
import torch

import distributed
from models.reporter_cls import StatisticsCLS as Statistics
from models.reporter_cls import ReportMgrCLS
from models.logging import logger
from models.loss import ConentSelectionLossCompute
from models.tree_reader import tree_building, headlist_to_string


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    print('gpu_rank %d' % gpu_rank)

    report_manager = ReportMgrCLS(args.report_every, start_time=-1)
    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):

        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.loss = ConentSelectionLossCompute(self.args.sentence_modelling_for_ext)

        assert grad_accum_count > 0
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed.all_gather_list(normalization))

                        self._gradient_accumulation(true_batchs, normalization, total_stats, report_stats)

                        report_stats = self._maybe_report_training(step, train_steps,
                                                                   [self.optim.learning_rate],
                                                                   report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats


    def validate(self, valid_iter, step=0):
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                mask_src = batch.mask_src
                tgt = batch.tgt
                mask_tgt = batch.mask_tgt
                clss = batch.clss
                mask_cls = batch.mask_cls
                labels = batch.gt_selection

                sent_scores, mask, _, _ = self.model(src, tgt, mask_src, mask_tgt, clss, mask_cls)

                loss = self.loss._compute_loss_test(labels, sent_scores, mask)
                loss = (loss * mask.float()).sum()

                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)

            self._report_step(0, step, valid_stats=stats)
            return stats


    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        src_path = '%s.raw_src' % (self.args.result_path)
        can_path = '%s.candidate' % (self.args.result_path)
        gold_path = '%s.gold' % (self.args.result_path)
        gold_select_path = '%s.gold_select' % (self.args.result_path)
        selected_ids_path = '%s.selected_ids' % (self.args.result_path)
        tree_path = '%s.trees' % (self.args.result_path)
        edge_path = '%s.edge' % (self.args.result_path)

        save_src = open(src_path, 'w')
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')
        save_gold_select = open(gold_select_path, 'w')
        save_selected_ids = open(selected_ids_path, 'w')
        save_trees = open(tree_path, 'w')
        save_edges = open(edge_path, 'w')

        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                mask_src = batch.mask_src
                tgt = batch.tgt
                mask_tgt = batch.mask_tgt
                clss = batch.clss
                mask_cls = batch.mask_cls
                labels = batch.gt_selection
                nsent = batch.nsent

                gold = []; pred = []; pred_select = []

                if (cal_lead):
                    selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                elif (cal_oracle):
                    selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                    range(batch.batch_size)]
                else:
                    sent_scores, mask, aj_matrixes, src_features = self.model(src, tgt, mask_src, mask_tgt, clss, mask_cls)

                    if (self.args.sentence_modelling_for_ext == 'tree'):
                        device = mask.device
                        sent_scores = sent_scores[-1] + mask.float()
                    else:
                        sent_scores = sent_scores+mask.float()

                    sent_scores = sent_scores.cpu().data.numpy()
                    selected_ids = np.argsort(-sent_scores, 1)
                    #selected_ids = np.sort(selected_ids,1)

                sent_nums = torch.sum(mask_cls, 1)
                for i, idx in enumerate(selected_ids):
                    _pred = []; _pred_select = []
                    if (len(batch.src_str[i]) == 0):
                        continue
                    sent_num = sent_nums[i]
                    if self.args.select_topn == 0:
                        select_topn = nsent[i]
                    elif self.args.select_topn > 0 and self.args.select_topn < 1:
                        select_topn = int(sent_num * self.args.select_topn) + 1
                    else:
                        select_topn = int(self.args.select_topn)
                    for j in selected_ids[i][:len(batch.src_str[i])]:
                        if (j >= len(batch.src_str[i])):
                            continue
                        candidate = batch.src_str[i][j].strip()
                        if (self.args.block_trigram):
                            if (not _block_tri(candidate, _pred)):
                                _pred.append(candidate)
                                _pred_select.append(int(j))
                        else:
                            _pred.append(candidate)
                            _pred_select.append(int(j))
                        if (not cal_oracle) and (not self.args.recall_eval) and len(_pred) == select_topn:
                            break

                    _pred = ' <q> '.join(_pred)

                    pred_select.append(_pred_select)
                    pred.append(_pred)
                    gold.append(' <q> '.join(batch.tgt_str[i]))

                selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in range(batch.batch_size)]

                for i in range(len(gold)):
                    save_gold.write(gold[i].strip()+'\n')
                for i in range(len(pred)):
                    save_pred.write(pred[i].strip()+'\n')
                for i in range(len(batch.src_str)):
                    save_src.write(' '.join(batch.src_str[i]).strip() + '\n')
                for i in range(len(pred_select)):
                    item = {'gold':selected_ids[i], 'pred':pred_select[i]}
                    save_selected_ids.write(json.dumps(item) + '\n')
                for i, idx in enumerate(selected_ids):
                    _gold_selection = [batch.src_str[i][j].strip() for j in selected_ids[i][:len(batch.src_str[i])]]
                    _gold_selection = ' <q> '.join(_gold_selection).strip()
                    save_gold_select.write(_gold_selection + '\n')

        self._report_step(0, step, valid_stats=stats)
        save_src.close()
        save_gold.close()
        save_pred.close()
        save_gold_select.close()
        save_selected_ids.close()
        save_trees.close()
        save_edges.close()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):

        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            mask_src = batch.mask_src
            verdict_labels = batch.verdict_labels
            pros_labels = batch.pros_labels
            cons_labels = batch.cons_labels
            cluster_sizes = batch.cluster_sizes

            verd_scores, pros_scores, cons_scores, cluster_masks = self.model(src, mask_src, cluster_sizes)

            loss_verd = self.loss._compute_loss(verdict_labels, verd_scores, cluster_masks)
            loss_pros = self.loss._compute_loss(pros_labels, pros_scores, cluster_masks)
            loss_cons = self.loss._compute_loss(cons_labels, cons_scores, cluster_masks)
            loss = loss_verd + loss_pros + loss_cons
            (loss / loss.numel()).backward()

            '''
            gradient_monitor = {}
            for name, para in self.model.named_parameters():
                if para.grad is not None:
                    print (torch.mean(para.grad))
            '''

            batch_stats = Statistics(float(loss.cpu().data.numpy()),
                                     normalization,
                                     loss_verd=float(loss_verd.cpu().data.numpy()),
                                     loss_pros=float(loss_pros.cpu().data.numpy()),
                                     loss_cons=float(loss_cons.cpu().data.numpy()))

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()


    def _save(self, step):
        real_model = self.model
        model_state_dict = real_model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'opt': self.args,
            'optims': [self.optim],
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
