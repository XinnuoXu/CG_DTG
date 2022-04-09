#!/bin/bash

BASE_DIR=./outputs.webnlg/

ADD_TOKEN_PATH=${BASE_DIR}/webnlg_toyproblem/predicates.txt
JSON_PATH=${BASE_DIR}/jsons/
BERT_DATA_PATH=${BASE_DIR}/data/
LOG_PATH=${BASE_DIR}/logs/

# Setup for WebNLG
#ADD_TOKEN_PATH=/home/s1687314/Planning/Plan_while_Generate/D2T_data/webnlg_toyproblem/predicates.txt
#JSON_PATH=/disk/scratch/s1687314/Planning/webnlg/outputs.webnlg/jsons/
#BERT_DATA_PATH=/disk/scratch/s1687314/Planning/webnlg/outputs.webnlg/data/
#LOG_PATH=/disk/scratch/s1687314/Planning/webnlg/outputs.webnlg/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${BERT_DATA_PATH}
rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-tokenizer ./outputs.cnn_dm/tokenizer.pt/ \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log
