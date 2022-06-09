import os, json

import numpy as np
import torch

import distributed
from models.reporter_abs import ReportMgr, Statistics
from models.reporter_ext import ReportMgrExt, StatisticsExt
from models.logging import logger
from tool.debug_tool import parameter_reporter

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims, loss, ext_loss=None):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    print('gpu_rank %d' % gpu_rank)

    trainer = Trainer(args, model, optims, loss, ext_loss, grad_accum_count, n_gpu, gpu_rank)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):

    def __init__(self,  args, model,  optims, loss, ext_loss=None,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank

        self.loss = loss
        self.ext_loss = ext_loss

        self.report_manager = ReportMgr(args.report_every, start_time=-1)
        self.report_manager_ext = ReportMgrExt(args.report_every, start_time=-1)

        if args.log_gradient == '':
            self.log_gradient = None
        else:
            self.log_gradient = open(args.log_gradient, 'w')

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()


    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step =  self.optims[0]._step + 1

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
                    num_tokens = batch.tgt[:, 1:].ne(self.loss.padding_idx).sum()
                    normalization += num_tokens.item()
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            self.report_manager,
                            step, train_steps,
                            [self.optims[i].learning_rate for i in range(len(self.optims))],
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


    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):

        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            mask_src = batch.mask_src
            mask_src_sent = batch.mask_src_sent
            mask_src_predicate = batch.src_predicate_mask
            tgt = batch.tgt
            mask_tgt = batch.mask_tgt
            clss = batch.clss
            mask_cls = batch.mask_cls
            labels = batch.alg
            gt_aj_matrix = batch.gt_aj_matrix
            prompt_tokenized = batch.prompt_tokenized

            outputs = self.model(src, tgt, mask_src, mask_tgt, 
                                 clss=clss, mask_cls=mask_cls, 
                                 labels=labels, 
                                 gt_aj_matrix=gt_aj_matrix,
                                 mask_src_sent=mask_src_sent,
                                 mask_src_predicate=mask_src_predicate,
                                 prompt_tokenized=prompt_tokenized)

            batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization)
            batch_stats.n_docs = int(src.size(0))

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if self.log_gradient is not None:
                gradients = parameter_reporter(self.model)
                self.log_gradient.write(json.dumps(gradients)+'\n')

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))
                for o in self.optims:
                    o.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            for o in self.optims:
                o.step()


    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                src = batch.src
                mask_src = batch.mask_src
                mask_src_sent = batch.mask_src_sent
                mask_src_predicate = batch.src_predicate_mask
                tgt = batch.tgt
                mask_tgt = batch.mask_tgt
                clss = batch.clss
                mask_cls = batch.mask_cls
                labels = batch.alg
                gt_aj_matrix = batch.gt_aj_matrix
                prompt_tokenized = batch.prompt_tokenized

                outputs = self.model(src, tgt, mask_src, mask_tgt, 
                                     clss=clss, mask_cls=mask_cls, 
                                     labels=labels, 
                                     gt_aj_matrix=gt_aj_matrix,
                                     mask_src_sent=mask_src_sent,
                                     mask_src_predicate=mask_src_predicate,
                                     prompt_tokenized=prompt_tokenized)

                batch_stats = self.loss.monolithic_compute_loss(batch, outputs)
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats


    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path


    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

        if self.report_manager_ext is not None:
            if start_time is None:
                self.report_manager_ext.start()
            else:
                self.report_manager_ext.start_time = start_time


    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, report_manager, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if report_manager is not None:
            return report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
