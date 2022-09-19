#!/bin/bash

BASE_DIR=./outputs.ama/

RAW_PATH=../Plan_while_Generate/AmaSum/AmaSum_data/
OUTPUT_PATH=${BASE_DIR}/raw_split/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}
rm -rf ${OUTPUT_PATH}/*

python preprocess.py \
	-mode simple_split_shard \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log
