#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division
import os
import argparse
from models.logging import init_logger
from train_abstractive import train_abs, validate_abs, test_abs
from train_reinforce import train_re, validate_re, test_re

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
    parser.add_argument("-ext_or_abs", default='abs', type=str, choices=['ext', 'abs', 'cls', 'slot', 'reinforce', 'slotsumm'])
    parser.add_argument("-cls_type", default='version_1', type=str)

    parser.add_argument("-input_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-log_gradient', default='')
    parser.add_argument('-load_pretrained_model', default='')

    # data parameters
    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-max_pos", default=1024, type=int)
    parser.add_argument("-max_tgt_len", default=250, type=int)
    parser.add_argument("-min_tgt_len", default=5, type=int)
    parser.add_argument("-max_src_nsent", default=2000, type=int)

    # parameters for extractive models and tmt
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=3, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    # planning parameters
    parser.add_argument("-cls_verdict", default='madeupword0000', type=str)
    parser.add_argument("-cls_pros", default='madeupword0001', type=str)
    parser.add_argument("-cls_cons", default='madeupword0002', type=str)
    parser.add_argument("-pred_special_tok", default='<PRED>', type=str)
    parser.add_argument("-obj_special_tok", default='<OBJ>', type=str)
    parser.add_argument("-slot_num_slots", default=7, type=int)
    parser.add_argument("-slot_iters", default=3, type=int)
    parser.add_argument("-slot_eps", default=1e-8, type=float)
    parser.add_argument("-slot_sample_mode", type=str, default='normal', choices=['marginal', 'normal', 'full_sample'])
    parser.add_argument("-slot_sample_schedule", type=str2bool, default=False)
    parser.add_argument("-cluster_algorithm", type=str, default='slot_attn', choices=['slot_attn', 'soft_kmeans'])
    parser.add_argument("-slotsumm_train_stage", type=str, default='pre-train', choices=['pre-train', 'gold_align', 'slot_attn'])

    parser.add_argument("-shuffle_src", type=str2bool, default=False)
    parser.add_argument("-conditional_decoder", type=str2bool, default=False)
    parser.add_argument("-pretrain_nn_cls", type=str2bool, default=False)
    parser.add_argument("-nn_cls_add_negative_samples", type=int, default=0)
    parser.add_argument("-pretrain_encoder_decoder", type=str2bool, default=False)
    parser.add_argument("-train_predicate_graph_only", type=str2bool, default=False)
    parser.add_argument("-gold_random_ratio", default=0.0, type=float)
    parser.add_argument("-spectral_ratio", default=0.0, type=float)
    parser.add_argument("-calculate_graph_prob_method", type=str, default='avg', choices=['avg', 'min'])
    parser.add_argument("-deterministic_graph_path", default='../Plan_while_Generate/D2T_data/webnlg_data/train.jsonl', type=str)
    parser.add_argument("-seen_predicate_tokenized_paths", default='', type=str)
    parser.add_argument("-from_scratch", type=str2bool, default=False)
    parser.add_argument("-spectral_with_sample", type=str2bool, default=False)
    parser.add_argument("-reinforce_strong_baseline", type=str2bool, default=False)
    parser.add_argument("-reinforce_threshold_baseline", type=str2bool, default=False)
    parser.add_argument("-reinforce_baseline_adja_threshold", default=-1, type=float)
    parser.add_argument("-reinforce_bernoulli_temp", default=2.0, type=float)
    parser.add_argument("-nn_graph", type=str2bool, default=False)
    parser.add_argument("-nn_graph_d_model", default=128, type=int)
    parser.add_argument("-nn_graph_d_ff", default=256, type=int)
    parser.add_argument("-nn_graph_heads", default=4, type=int)
    parser.add_argument("-nn_graph_nlayers", default=2, type=int)
    parser.add_argument("-nn_graph_dropout", default=0.1, type=float)
    parser.add_argument("-test_alignment_type", type=str, default='spectral', choices=['gold', 'spectral', 'discriministic', 'full_src', 'random_test'])
    parser.add_argument("-test_given_nclusters", type=str2bool, default=True)
    parser.add_argument("-test_graph_selection_threshold", default=1.0, type=float)
    parser.add_argument("-test_entity_link", type=str2bool, default=False)
    parser.add_argument("-test_no_single_pred_score", type=str2bool, default=False)
    parser.add_argument("-test_unseen", type=str2bool, default=False)
    parser.add_argument("-test_run_bernoulli", type=str2bool, default=True)
    parser.add_argument("-test_adja_threshold", default=-1, type=float)

    # generation parameters
    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)

    # traning parameters
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-lr_planner", default=-1, type=float)
    parser.add_argument("-lr_encdec", default=-1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-decay_method", default='noam', type=str)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_planner", default=8000, type=int)
    parser.add_argument("-warmup_steps_encdec", default=8000, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument('-seed', default=42, type=int) # used to be 666
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-master_port', default='10000', type=str)
    parser.add_argument("-train_from", default='')
    parser.add_argument("-reset_optimizer", type=str2bool, default=False)
    parser.add_argument("-load_from_ext", default='')
    parser.add_argument("-load_from_abs", default='')

    # test parameters
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-select_topn", default=3, type=float)
    parser.add_argument("-test_data_source", default='test', type=str, choices=['train', 'validation', 'test'])
    parser.add_argument("-test_min_length", default=10, type=int)
    parser.add_argument("-test_max_length", default=60, type=int)
    parser.add_argument("-test_lang", default='en', type=str)
    parser.add_argument("-test_no_repeat_ngram_size", default=0, type=int)
    parser.add_argument("-test_length_penalty", default=1.0, type=float)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if args.ext_or_abs == 'reinforce':
        if (args.mode == 'train'):
            train_re(args, device_id)
        elif (args.mode == 'validate'):
            validate_re(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_re(args, device_id, cp, step)

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
