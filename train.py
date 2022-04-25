#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import os
import argparse
from models.logging import init_logger
from train_extractive import train_ext, validate_ext, test_ext
from train_abstractive import train_abs, validate_abs, test_abs
from train_mixture import train_mix, validate_mix, test_mix
from train_stepwise import train_stepwise, validate_stepwise, test_stepwise
from train_treeabs import train_treeabs, validate_treeabs, test_treeabs

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", default='facebook/bart-base', type=str)
    parser.add_argument("-tokenizer_path", default='facebook/bart-base', type=str)
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-ext_or_abs", default='abs', type=str, choices=['ext', 'abs', 'mix', 'step', 'plan'])
    parser.add_argument("-inference_mode", default='abs', type=str, choices=['abs', 'non_prjective_tree'])
    parser.add_argument("-content_planning_model", default='', type=str, choices=['transformer', 'tree', 'none'])

    parser.add_argument("-input_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')
    parser.add_argument('-log_file', default='../logs/cnndm.log')

    # data parameters
    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-max_pos", default=1024, type=int)
    parser.add_argument("-max_tgt_len", default=250, type=int)

    # planning parameters
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=3, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)
    parser.add_argument("-tree_gumbel_softmax_tau", default=0.3, type=float)
    parser.add_argument("-freeze_encoder_decoder", type=str2bool, default=False)
    parser.add_argument("-freeze_tmt", type=str2bool, default=False)
    parser.add_argument("-planning_method", type=str, default='gumbel_tree', choices=['gumbel_tree', 'topk_tree', 'ground_truth', 'random', 'lead_k', 'not_lead_k'])
    parser.add_argument("-ext_topn", default=3, type=float)

    # generation parameters
    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)

    # traning parameters
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-lr_tmt", default=-1, type=float)
    parser.add_argument("-lr_enc_dec", default=-1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-decay_method", default='noam', type=str)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_tmt", default=8000, type=int)
    parser.add_argument("-warmup_steps_enc_dec", default=8000, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument('-seed', default=666, type=int)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument("-train_from", default='')
    parser.add_argument("-load_from_ext", default='')
    parser.add_argument("-load_from_abs", default='')
    parser.add_argument("-abs_plus_ext_loss", type=float, nargs='?', default=0.0)

    # test parameters
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-select_topn", default=3, type=float)
    parser.add_argument("-test_data_source", default='test', type=str, choices=['train', 'validation', 'test'])
    parser.add_argument("-test_min_length", default=10, type=int)
    parser.add_argument("-test_max_length", default=60, type=int)
    parser.add_argument("-do_analysis", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1


    if args.ext_or_abs == 'ext':
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)

    elif args.ext_or_abs == 'abs':
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)

    elif args.ext_or_abs == 'mix':
        if (args.mode == 'train'):
            train_mix(args, device_id)
        elif (args.mode == 'validate'):
            validate_mix(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_mix(args, device_id, cp, step)

    elif args.ext_or_abs == 'step':
        if (args.mode == 'train'):
            train_stepwise(args, device_id)
        elif (args.mode == 'validate'):
            validate_stepwise(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_stepwise(args, device_id, cp, step)

    elif args.ext_or_abs == 'plan':
        if (args.mode == 'train'):
            train_treeabs(args, device_id)
        elif (args.mode == 'validate'):
            validate_treeabs(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_treeabs(args, device_id, cp, step)
