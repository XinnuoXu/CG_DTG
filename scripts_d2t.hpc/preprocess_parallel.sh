#!/bin/bash

# Setup for WebNLG
ADD_TOKEN_PATH=../Plan_while_Generate/D2T_data/webnlg_data/predicates.txt
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
JSON_PATH=${BASE_PATH}/jsons.aug/
DATA_PATH=${BASE_PATH}/data.parallel/
LOG_PATH=${BASE_PATH}/logs.parallel/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
rm -rf ${DATA_PATH}/*

python preprocess.py \
	-mode format_for_parallel \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-n_cpus 32 \
	-tokenizer t5-base \
        -max_src_ntokens 512 \
        -max_tgt_ntokens 512 \
	-shuffle_src True \
	-log_file ${LOG_PATH}/preprocess.log
	
