#!/bin/bash

#JSON_PATH=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/jsons/
BASE_DIR=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/
JSON_PATH=${BASE_DIR}/jsons/
BERT_DATA_PATH=${BASE_DIR}/data/
LOG_PATH=${BASE_DIR}/logs/

mkdir ${BERT_DATA_PATH}
#rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-tokenizer facebook/bart-base \
	-n_cpus 1 \
	-log_file ${LOG_PATH}/preprocess.log
