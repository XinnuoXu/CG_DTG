#encoding=utf-8

import argparse
import time
from models.logging import init_logger
from prepro import data_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", default='', type=str)
    parser.add_argument('-dataset', default='')
    parser.add_argument("-raw_path", default='')
    parser.add_argument("-test_selection_path", default='')
    parser.add_argument("-test_sentiment_path", default='')
    parser.add_argument("-save_path", default='')
    parser.add_argument("-additional_token_path", default='')
    parser.add_argument("-saved_tokenizer_path", default='')
    parser.add_argument("-seen_predicate_paths", default='../Plan_while_Generate/D2T_data/webnlg_data/predicates_seen.txt', type=str)
    parser.add_argument("-seen_predicate_tokenized_paths", default='', type=str)
    parser.add_argument("-predicted_plan_path", default='')
    parser.add_argument("-predicted_plan_id_path", default='')
    parser.add_argument('-log_file', default='./logs/cnndm.log')
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    parser.add_argument('-max_src_ntokens', default=1024, type=int)
    parser.add_argument("-max_cluster_num", default=250, type=int)
    parser.add_argument("-oracle_topn", default=-1, type=int)
    parser.add_argument("-tokenizer", default='facebook/bart-base')
    parser.add_argument("-add_plan_to_tgt", type=str, default='none', choices=['prompt', 'intersec', 'none'])
    parser.add_argument("-add_plan_to_src", type=str, default='none', choices=['none', 'hard_prompt', 'soft_prompt'])
    parser.add_argument("-for_stepwise", type=str2bool, default=False)
    parser.add_argument("-plan_generation", type=str2bool, default=False)
    parser.add_argument("-no_bos_for_tgt", type=str2bool, default=False)
    parser.add_argument("-shuffle_src", type=str2bool, default=False)
    parser.add_argument("-multi_ref_test", type=str2bool, default=False)
    parser.add_argument("-tokenize_src_predicate", type=str2bool, default=False)

    parser.add_argument("-spectral_method", default='spectral_clustering', type=str, choices=['spectral_clustering', 'random', 'only_entity', 'ground_truth'])
    parser.add_argument("-spectral_assign_labels", default='discretize', type=str)
    parser.add_argument("-spectral_eigen_solver", default='arpack', type=str)
    parser.add_argument("-spectral_affinity", default='precomputed', type=str)
    parser.add_argument("-spectral_max_group_size", default=3, type=int)
    parser.add_argument("-spectral_min_pair_freq", default=20, type=int)
    parser.add_argument("-spectral_use_ratio", type=str2bool, default=False)
    parser.add_argument("-spectral_filter_with_entities", type=str2bool, default=True)
    parser.add_argument("-spectral_train_file", default='../Plan_while_Generate/D2T_data/webnlg_data/train.jsonl', type=str)
    parser.add_argument("-remove_noise_datapoints", type=str2bool, default=False)
    parser.add_argument("-remove_single_triple_datapoints", type=str2bool, default=False)

    parser.add_argument("-amasum_verdict_cluster_topk", default=3, type=int)
    parser.add_argument("-amasum_pros_cluster_topk", default=6, type=int)
    parser.add_argument("-amasum_cons_cluster_topk", default=3, type=int)
    parser.add_argument("-amasum_verdict_sentiment_threshold", default=0.3, type=float)
    parser.add_argument("-amasum_pros_sentiment_threshold", default=0.7, type=float)
    parser.add_argument("-amasum_cons_sentiment_threshold", default=0.3, type=float)
    parser.add_argument("-amasum_binary_selection_topk", default=12, type=int)
    parser.add_argument("-amasum_binary_selection_threshold", default=0.05, type=float)
    parser.add_argument("-amasum_classification_threshold", default=0.0, type=float)
    parser.add_argument("-amasum_random_topk", default=False, type=str2bool)
    parser.add_argument("-amasum_delete_empty_example", default=False, type=str2bool)
    parser.add_argument("-amasum_sentiment_cls", default=False, type=str2bool)

    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.'+args.mode + '(args)')
