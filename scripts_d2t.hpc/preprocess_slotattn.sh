#!/bin/bash

# Setup for WebNLG
ADD_TOKEN_PATH=../Plan_while_Generate/D2T_data/webnlg_data/predicates.txt
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
#JSON_PATH=${BASE_PATH}/jsons/
#DATA_PATH=${BASE_PATH}/data/
#LOG_PATH=${BASE_PATH}/logs/
JSON_PATH=${BASE_PATH}/test.jsons/
DATA_PATH=${BASE_PATH}/test.data/
LOG_PATH=${BASE_PATH}/test.logs/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
rm -rf ${DATA_PATH}/*

python preprocess.py \
	-mode format_for_slot_attn \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-n_cpus 32 \
	-tokenizer t5-base \
	-shuffle_src True \
	-log_file ${LOG_PATH}/preprocess.log
	
