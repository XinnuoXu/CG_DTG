#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import json

import torch
import distributed
from transformers import AutoTokenizer
from models import data_slotsumm_loader, model_builder
from models.data_slotsumm_loader import load_dataset
from models.loss import abs_loss 
from models.model_builder import SlotSumm
from models.trainer_slotsumm import build_trainer
#from models.predictor_slotsumm import build_predictor
from models.logging import logger, init_logger

model_flags = ['model_name', 'ext_or_abs', 'tokenizer_path', 
               'ext_layers', 'ext_heads', 'ext_ff_size', 'use_mask_in_loss']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks, args.master_port)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_slotsumm_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def train_slotsumm(args, device_id):
    if (args.world_size > 1):
        train_slotsumm_multi(args)
    else:
        train_slotsumm_single(args, device_id)


def train_slotsumm_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args, device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def train_slotsumm_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    # Load checkpoint
    checkpoint = None
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_slotsumm_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), 
                                               args.batch_size, device, shuffle=True, is_test=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    pad_token_id = tokenizer.pad_token_id

    model = SlotSumm(args, device, len(tokenizer), pad_token_id, checkpoint)

    # Load optimizer
    if args.lr_encdec > -1 and args.lr_planner > -1:
        optim_encdec = model_builder.build_optim_encdec(args, model, checkpoint)
        optim_planner = model_builder.build_optim_planner(args, model, checkpoint)
        optim = [optim_encdec, optim_planner]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)

    # Load loss
    symbols = {'PAD': pad_token_id}
    train_loss = abs_loss(model.generator, model.vocab_size, device, 
                          symbols=symbols, train=True, 
                          label_smoothing=args.label_smoothing)
    # Load trainer
    trainer = build_trainer(args, device_id, model, optim, train_loss)

    # Start training
    trainer.train(train_iter_fct, args.train_steps)


def validate_slotsumm(args, device_id):
    timestep = 0
    if args.test_from != "":
        step = args.test_from.split('_')[-1].split('.')[0]
        validate(args, device_id, args.test_from, step)
    else:
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    valid_iter = data_slotsumm_loader.Dataloader(args, load_dataset(args, 'validation', shuffle=False),
                                             args.batch_size, device,
                                             shuffle=False, is_test=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    symbols = {'PAD': tokenizer.pad_token_id}

    if args.load_pretrained_model != '':
        pretrained_model_checkpoint = torch.load(args.load_pretrained_model, map_location=lambda storage, loc: storage)
        pretrained_model = AbsSummarizer(args, device, tokenizer.cls_token_id, len(tokenizer), pretrained_model_checkpoint)
    else:
        pretrained_model = None

    if args.slot_sample_mode == 'full_sample':
        model = SlotAttnAggragatorDiscrete(args, device, len(tokenizer), checkpoint, pretrained_checkpoint=pretrained_model)
    else:
        model = SlotAttnAggragator(args, device, len(tokenizer), checkpoint, pretrained_checkpoint=pretrained_model)

    model.eval()

    valid_loss = abs_loss(model.generator, model.vocab_size, 
                          symbols=symbols, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)

    return stats.xent()


def test_slotsumm(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    test_iter = data_slotsumm_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                            args.test_batch_size, device,
                                            shuffle=False, is_test=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.load_pretrained_model != '':
        pretrained_model_checkpoint = torch.load(args.load_pretrained_model, map_location=lambda storage, loc: storage)
        #pretrained_model = AbsSummarizer(args, device, tokenizer.cls_token_id, len(tokenizer), pretrained_model_checkpoint)
        pretrained_model = SlotAttnAggragator(args, device, len(tokenizer), pretrained_model_checkpoint)
    else:
        pretrained_model = None

    if args.slot_sample_mode == 'full_sample':
        model = SlotAttnAggragatorDiscrete(args, device, len(tokenizer), checkpoint, pretrained_checkpoint=pretrained_model)
    else:
        model = SlotAttnAggragator(args, device, len(tokenizer), checkpoint, pretrained_checkpoint=pretrained_model)

    model.eval()

    #predictor = build_predictor(args, tokenizer, model, logger)
    #predictor.translate(test_iter, step)


