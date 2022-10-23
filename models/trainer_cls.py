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
        self.loss = ConentSelectionLossCompute()

        assert grad_accum_count > 0
        if (model):
            self.model.train()


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

            loss_list = []
            if self.args.cls_type == 'select':
                verdict_labels = (verdict_labels + pros_labels + cons_labels)
                verdict_labels = (verdict_labels > 0)
            if verd_scores is not None:
                loss_verd = self.loss._compute_loss(verdict_labels, verd_scores, cluster_masks)
                loss_list.append(loss_verd)
            if pros_scores is not None:
                loss_pros = self.loss._compute_loss(pros_labels, pros_scores, cluster_masks)
                loss_list.append(loss_pros)
            if cons_scores is not None:
                loss_cons = self.loss._compute_loss(cons_labels, cons_scores, cluster_masks)
                loss_list.append(loss_cons)
            loss = sum(loss_list) #loss = loss_verd + loss_pros + loss_cons
            (loss / loss.numel()).backward()

            '''
            batch_stats = Statistics(float(loss.cpu().data.numpy()),
                                     normalization,
                                     loss_verd=float(loss_verd.cpu().data.numpy()),
                                     loss_pros=float(loss_pros.cpu().data.numpy()),
                                     loss_cons=float(loss_cons.cpu().data.numpy()))
            '''
            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

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
                verdict_labels = batch.verdict_labels
                pros_labels = batch.pros_labels
                cons_labels = batch.cons_labels
                cluster_sizes = batch.cluster_sizes

                verd_scores, pros_scores, cons_scores, cluster_masks = self.model(src, mask_src, cluster_sizes)

                loss_list = []
                if self.args.cls_type == 'select':
                    verdict_labels = (verdict_labels + pros_labels + cons_labels)
                    verdict_labels = (verdict_labels > 0)
                if verd_scores is not None:
                    loss_verd = self.loss._compute_loss(verdict_labels, verd_scores, cluster_masks)
                    loss_list.append(loss_verd)
                if pros_scores is not None:
                    loss_pros = self.loss._compute_loss(pros_labels, pros_scores, cluster_masks)
                    loss_list.append(loss_pros)
                if cons_scores is not None:
                    loss_cons = self.loss._compute_loss(cons_labels, cons_scores, cluster_masks)
                    loss_list.append(loss_cons)
                loss = sum(loss_list) #loss = loss_verd + loss_pros + loss_cons

                '''
                batch_stats = Statistics(float(loss.cpu().data.numpy()),
                                         normalization,
                                         loss_verd=float(loss_verd.cpu().data.numpy()),
                                         loss_pros=float(loss_pros.cpu().data.numpy()),
                                         loss_cons=float(loss_cons.cpu().data.numpy()))
                '''
                batch_stats = Statistics(float(loss.cpu().data.numpy()), src.size(0))

                stats.update(batch_stats)

            self._report_step(0, step, valid_stats=stats)
            return stats


    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):

        self.model.eval()

        res_path = '%s.json' % (self.args.result_path)
        save_res = open(res_path, 'w')

        with torch.no_grad():
            for batch in test_iter:

                src = batch.src
                mask_src = batch.mask_src
                verdict_labels = batch.verdict_labels
                pros_labels = batch.pros_labels
                cons_labels = batch.cons_labels
                cluster_sizes = batch.cluster_sizes
                clusters = batch.clusters
                eid = batch.eid

                verd_scores, pros_scores, cons_scores, cluster_masks = self.model(src, mask_src, cluster_sizes)

                if self.args.cls_type == 'select':
                    verdict_labels = (verdict_labels + pros_labels + cons_labels)
                    verdict_labels = (verdict_labels > 0)

                if verd_scores is not None:
                    verd_scores = verd_scores.cpu().data.numpy()
                else:
                    verd_scores = np.zeros(verdict_labels.size())

                if pros_scores is not None:
                    pros_scores = pros_scores.cpu().data.numpy()
                else:
                    pros_scores = np.zeros(pros_labels.size())

                if cons_scores is not None:
                    cons_scores = cons_scores.cpu().data.numpy()
                else:
                    cons_scores = np.zeros(cons_labels.size())

                sent_nums = [len(cluster_size) for cluster_size in cluster_sizes]

                for i in range(len(sent_nums)):
                    sent_num = sent_nums[i]

                    json_obj = {}
                    json_obj['clusters'] = clusters[i]
                    json_obj['verdict_labels'] = verdict_labels[i][:sent_num].cpu().data.numpy().tolist()
                    json_obj['pros_labels'] = pros_labels[i][:sent_num].cpu().data.numpy().tolist()
                    json_obj['cons_labels'] = cons_labels[i][:sent_num].cpu().data.numpy().tolist()
                    
                    json_obj['verdict_scores'] = verd_scores[i][:sent_num].tolist()
                    json_obj['pros_scores'] = pros_scores[i][:sent_num].tolist()
                    json_obj['cons_scores'] = cons_scores[i][:sent_num].tolist()
                    json_obj['example_id'] = eid[i]
                    
                    save_res.write(json.dumps(json_obj)+'\n')

        save_res.close()


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
