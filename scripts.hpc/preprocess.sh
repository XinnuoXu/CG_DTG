#!/bin/bash

# Setup for Xsum
JSON_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/jsons/
BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs/

# Setup for CNN
#JSON_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/jsons/
#BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/data/
#LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${BERT_DATA_PATH}
#rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-tokenizer facebook/bart-base \
	-n_cpus 1 \
	-log_file ${LOG_PATH}/preprocess.log
