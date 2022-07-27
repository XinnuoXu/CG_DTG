#!/bin/bash

# Setup for Webnlg
BASE_DIR=./outputs.webnlg/

RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_partial_src/
JSON_PATH=${BASE_DIR}/jsons.partial_src/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
        -mode split_shard \
        -dataset train \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log \

python preprocess.py \
        -mode split_shard \
        -dataset validation \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log \


RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_data/
JSON_PATH=${BASE_DIR}/jsons.partial_src/
LOG_PATH=${BASE_DIR}/logs/

python preprocess.py \
        -mode split_shard_spectral_cluster \
        -dataset test \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -spectral_train_file ${RAW_PATH}/train.jsonl \
        -spectral_use_ratio False \
        -spectral_filter_with_entities True \
        -spectral_min_pair_freq 20 \
        -spectral_max_group_size 3 \
        -log_file ${LOG_PATH}/preprocess_shard.log \
