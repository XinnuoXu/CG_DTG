#!/bin/bash

RAW_PATH=/home/hpcxu1/Planning//Plan_while_Generate/Content_Selection/xsum_origin_greedy_sent.oracle/
JSON_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/jsons/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs/

mkdir ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
	-mode split_shard \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-oracle_topn 1000 \
	-n_cpus 30 \
	-log_file ${LOG_PATH}/preprocess_shard.log \
