#!/bin/bash

# Setup for Xsum
#JSON_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/jsons/
#BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
#LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs/

# Setup for CNN
BASE_DIR=./outputs.cnn_dm/
RAW_PATH=../Plan_while_Generate/Content_Selection/cnn_origin_greedy_sent.entity_chain/

ADD_TOKEN_PATH=${RAW_PATH}/special_tokens.txt
JSON_PATH=${BASE_DIR}/jsons/
BERT_DATA_PATH=${BASE_DIR}/data/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${BERT_DATA_PATH}
rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-tokenizer facebook/bart-base \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log
