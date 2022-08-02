#!/bin/bash

BASE_DIR=./outputs.webnlg/

RAW_PATH=../Plan_while_Generate/D2T_data/
ADD_TOKEN_PATH=${RAW_PATH}/webnlg_data/predicates.txt
JSON_PATH=${BASE_DIR}/jsons.prefix_tgt/
BERT_DATA_PATH=${BASE_DIR}/data.prefix_tgt/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${BERT_DATA_PATH}
rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
        -dataset train \
        -tokenizer t5-small \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-n_cpus 30 \
	-log_file ${LOG_PATH}/preprocess.log

python preprocess.py \
	-mode format_for_training \
        -dataset validation \
        -tokenizer t5-small \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-n_cpus 30 \
	-log_file ${LOG_PATH}/preprocess.log

python preprocess.py \
	-mode format_for_prefix_tgt_test \
        -dataset test \
        -tokenizer t5-small \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-n_cpus 30 \
	-log_file ${LOG_PATH}/preprocess.log

