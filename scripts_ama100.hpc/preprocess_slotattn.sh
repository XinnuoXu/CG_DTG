#!/bin/bash

# Setup for WebNLG
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.ama100/
JSON_PATH=${BASE_PATH}/raw_split/
DATA_PATH=${BASE_PATH}/test.data/
LOG_PATH=${BASE_PATH}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
rm -rf ${DATA_PATH}/*

python preprocess.py \
	-mode format_for_slot_summ \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-max_src_ntokens 1024 \
	-max_tgt_ntokens 800 \
	-n_cpus 32 \
	-tokenizer facebook/bart-base \
	-log_file ${LOG_PATH}/preprocess.log
	
