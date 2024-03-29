import os, json
import random

import numpy as np
import torch

import distributed
from models.reporter_abs import ReportMgr, Statistics
from models.logging import logger
from tool.debug_tool import parameter_reporter

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optims, pad_id):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    print('gpu_rank %d' % gpu_rank)

    trainer = Trainer(args, model, optims, grad_accum_count, n_gpu, gpu_rank, pad_id)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):

    def __init__(self,  args, model, optims, grad_accum_count=1, n_gpu=1, gpu_rank=1, pad_id=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.pad_id = pad_id

        self.report_manager = ReportMgr(args.report_every, start_time=-1)

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
                    num_tokens = sum([tgt[:, 1:].ne(self.pad_id).sum() for tgt in batch.tgt])
                    normalization += num_tokens.item()
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(true_batchs, normalization, total_stats, report_stats, step)

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


    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats, step):

        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:

            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            tgt = batch.tgt
            mask_tgt = batch.mask_tgt
            ctgt = batch.ctgt
            mask_ctgt = batch.mask_ctgt
            mask_ctgt_loss = batch.mask_ctgt_loss
            preds = batch.pred
            pred_tokens = batch.pred_tokens
            pred_mask_tokens = batch.pred_mask_tokens
            aggregation_labels = batch.aggregation_labels
            p2s = batch.p2s
            nsent = batch.nsent

            bias = 0.0
            if self.args.pretrain_nn_cls:
                loss, logging_info = self.model(src, tgt, mask_tgt, 
                                                ctgt, mask_ctgt, mask_ctgt_loss, 
                                                preds, pred_tokens, pred_mask_tokens, 
                                                p2s, nsent,
                                                aggregation_labels=aggregation_labels,
                                                mode='nn')
                loss = loss.sum()
                (loss/loss.numel()).backward()

            elif self.args.pretrain_encoder_decoder:
                loss, weights, _, logging_info = self.model(src, tgt, mask_tgt, 
                                                        ctgt, mask_ctgt, mask_ctgt_loss, 
                                                        preds, pred_tokens, pred_mask_tokens, 
                                                        p2s, nsent, mode='gold')
                #loss = -cll.sum()
                loss.backward()

            else:
                # conditional log likelihood
                sampled_num = random.random()
                if sampled_num <= self.args.gold_random_ratio:
                    cll, weights, bias, logging_info = self.model(src, tgt, mask_tgt, 
                                                            ctgt, mask_ctgt, mask_ctgt_loss, 
                                                            preds, pred_tokens, pred_mask_tokens, 
                                                            p2s, nsent, mode='gold')
                elif sampled_num >= 1 - self.args.spectral_ratio:
                    cll, weights, bias, logging_info = self.model(src, tgt, mask_tgt, 
                                                            ctgt, mask_ctgt, mask_ctgt_loss, 
                                                            preds, pred_tokens, pred_mask_tokens, 
                                                            p2s, nsent, mode='spectral')
                else:
                    cll, weights, bias, logging_info = self.model(src, tgt, mask_tgt, 
                                                            ctgt, mask_ctgt, mask_ctgt_loss, 
                                                            preds, pred_tokens, pred_mask_tokens, 
                                                            p2s, nsent, mode='random')

                # baseline
                if self.args.reinforce_strong_baseline:
                    baseline_cll, _, _, _ = self.model(src, tgt, mask_tgt, 
                                                    ctgt, mask_ctgt, mask_ctgt_loss, 
                                                    preds, pred_tokens, pred_mask_tokens, 
                                                    p2s, nsent, mode='spectral_baseline') # baseline
                elif self.args.reinforce_threshold_baseline:
                    baseline_cll, _, _, _ = self.model(src, tgt, mask_tgt, 
                                                    ctgt, mask_ctgt, mask_ctgt_loss, 
                                                    preds, pred_tokens, pred_mask_tokens, 
                                                    p2s, nsent, mode='threshold_baseline')
                else:
                    baseline_cll, _, _, _ = self.model(src, tgt, mask_tgt, 
                                                    ctgt, mask_ctgt, mask_ctgt_loss, 
                                                    preds, pred_tokens, pred_mask_tokens, 
                                                    p2s, nsent, mode='random') # baseline

                # calculte loss
                rec = (weights * (cll - baseline_cll))
                loss = -rec
                loss = loss.sum() + sum(bias) * 0.1
                loss.backward()

                bias_log = sum(bias).clone().item()/len(bias)

            #parameter_reporter(self.model.abs_model)


            batch_stats = Statistics(loss.clone().item(), 
                                     logging_info['num_non_padding'], 
                                     logging_info['num_correct'],
                                     bias = bias_log)
            batch_stats.n_docs = logging_info['n_docs']

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

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
                tgt = batch.tgt
                mask_tgt = batch.mask_tgt
                ctgt = batch.ctgt
                mask_ctgt = batch.mask_ctgt
                mask_ctgt_loss = batch.mask_ctgt_loss
                preds = batch.pred
                pred_tokens = batch.pred_tokens
                pred_mask_tokens = batch.pred_mask_tokens
                aggregation_labels = batch.aggregation_labels
                p2s = batch.p2s
                nsent = batch.nsent

                if self.args.pretrain_nn_cls:
                    loss, logging_info = self.model(src, tgt, mask_tgt, 
                                                    ctgt, mask_ctgt, mask_ctgt_loss, 
                                                    preds, pred_tokens, pred_mask_tokens, 
                                                    p2s, nsent,
                                                    aggregation_labels=aggregation_labels,
                                                    mode='nn')
                    loss = loss.sum()
                    loss = loss/loss.numel()

                elif self.args.pretrain_encoder_decoder:
                    loss, weights, logging_info = self.model(src, tgt, mask_tgt, 
                                                            ctgt, mask_ctgt, mask_ctgt_loss, 
                                                            preds, pred_tokens, pred_mask_tokens, 
                                                            p2s, nsent, mode='gold')
                    #loss = -cll.sum()

                else:
                    cll, weights, bias, logging_info = self.model(src, tgt, mask_tgt, 
                                                            ctgt, mask_ctgt, mask_ctgt_loss, 
                                                            preds, pred_tokens, pred_mask_tokens, 
                                                            p2s, nsent, mode='spectral')
                    loss = -cll.sum()

                batch_stats = Statistics(loss.clone().item(), logging_info['num_non_padding'], logging_info['num_correct'])
                batch_stats.n_docs = logging_info['n_docs']
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
