#!/bin/bash

# Setup for Webnlg
BASE_DIR=./outputs.webnlg/

RAW_PATH=${BASE_DIR}/webnlg_data/
JSON_PATH=${BASE_DIR}/jsons/
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
