#!/bin/bash

# Setup for WebNLG
ADD_TOKEN_PATH=/home/hpcxu1/Planning/Plan_while_Generate/D2T_data/webnlg_data/predicates.txt
JSON_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/jsons/
BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/data.tmp/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${BERT_DATA_PATH}
rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-n_cpus 32 \
	-tokenizer t5-base \
	-log_file ${LOG_PATH}/preprocess.log
	
