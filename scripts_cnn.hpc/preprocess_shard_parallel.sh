#!/bin/bash

# Setup for XSum
#RAW_PATH=/home/hpcxu1/Planning//Plan_while_Generate/Content_Selection/xsum_origin_greedy_sent.oracle/
#JSON_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/jsons/
#LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs/

# Setup for CNN
RAW_PATH=../Plan_while_Generate/Content_Selection/cnn_origin_greedy_sent.sentences/

BASE_DIR=./outputs.cnn_dm/
JSON_PATH=${BASE_DIR}/jsons.parallel/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
        -mode split_shard \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log \
