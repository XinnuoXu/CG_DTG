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
import torch
import distributed
from models import model_builder, data_cls_loader
from models.data_cls_loader import load_dataset
from models.model_builder import ParagraphMultiClassifier
from models.model_builder import ClusterMultiClassifier
from models.model_builder import ClusterLongformerClassifier
from models.model_builder import ClusterBinarySelection
from models.model_builder import SentimentClassifier
from models.trainer_cls import build_trainer
from models.logging import logger, init_logger
from transformers import AutoTokenizer

model_flags = ['max_src_nsent', 'ext_ff_size', 'ext_heads', 'ext_dropout', 'ext_layers', 'ext_or_abs', 'cls_type', 'cls_verdict', 'cls_pros', 'cls_cons']


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


def validate_cls(args, device_id):
    timestep = 0
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    xent_lst = []
    for i, cp in enumerate(cp_files):
        step = int(cp.split('.')[-2].split('_')[-1])
        xent = validate(args, device_id, cp, step)
        xent_lst.append((xent, cp))
        max_step = xent_lst.index(min(xent_lst))
        if (i - max_step > 10):
            break
    xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.cls_type == 'version_1':
        model = ParagraphMultiClassifier(args, device, checkpoint)
    elif args.cls_type == 'version_2':
        model = ClusterMultiClassifier(args, device, checkpoint)
    elif args.cls_type == 'version_3':
        model = ClusterLongformerClassifier(args, device, tokenizer, checkpoint)
    elif args.cls_type in ['pros', 'cons', 'verdict', 'select']:
        model = ClusterBinarySelection(args, device, tokenizer, checkpoint)
    elif args.cls_type == 'sentiment_cls':
        model = SentimentClassifier(args, device, tokenizer, checkpoint)
    model.eval()

    valid_iter = data_cls_loader.Dataloader(args, load_dataset(args, 'validation', shuffle=False),
                                            args.batch_size, device, shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_cls(args, device_id, pt, step):
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.cls_type == 'version_1':
        model = ParagraphMultiClassifier(args, device, checkpoint)
    elif args.cls_type == 'version_2':
        model = ClusterMultiClassifier(args, device, checkpoint)
    elif args.cls_type == 'version_3':
        model = ClusterLongformerClassifier(args, device, tokenizer, checkpoint)
    elif args.cls_type in ['pros', 'cons', 'verdict', 'select']:
        model = ClusterBinarySelection(args, device, tokenizer, checkpoint)
    elif args.cls_type == 'sentiment_cls':
        model = SentimentClassifier(args, device, tokenizer, checkpoint)
    model.eval()

    test_iter = data_cls_loader.Dataloader(args, load_dataset(args, args.test_data_source, shuffle=False),
                                           args.batch_size, device, shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)

    trainer.test(test_iter, step)


def train_multi_cls(args):
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


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks, args.master_port)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_cls(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


def train_single_cls(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_cls_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), 
                                          args.batch_size, device, shuffle=True, is_test=False)

    print ('TOKENIZER', args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.cls_type == 'version_1':
        model = ParagraphMultiClassifier(args, device, checkpoint)
    elif args.cls_type == 'version_2':
        model = ClusterMultiClassifier(args, device, checkpoint)
    elif args.cls_type == 'version_3':
        model = ClusterLongformerClassifier(args, device, tokenizer, checkpoint)
    elif args.cls_type in ['pros', 'cons', 'verdict', 'select']:
        model = ClusterBinarySelection(args, device, tokenizer, checkpoint)
    elif args.cls_type == 'sentiment_cls':
        model = SentimentClassifier(args, device, tokenizer, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)


def train_cls(args, device_id):
    if (args.world_size > 1):
        train_multi_cls(args)
    else:
        train_single_cls(args, device_id)


