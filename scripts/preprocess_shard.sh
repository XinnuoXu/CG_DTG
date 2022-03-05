#!/bin/bash

# RAW_DATA_NAME=xsum_origin_greedy_sent.oracle/
RAW_PATH=${BASE_DIR}/${RAW_DATA_NAME}/
JSON_PATH=${BASE_DIR}/jsons/
LOG_PATH=${BASE_DIR}/logs/

mkdir ${LOG_PATH}
mkdir ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
	-mode split_shard \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-oracle_topn 1000 \
	-n_cpus 30 \
	-log_file ${LOG_PATH}/preprocess_shard.log \
