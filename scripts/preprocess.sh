#!/bin/bash

#JSON_PATH=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/jsons/
#BERT_DATA_PATH=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/data/
JSON_PATH=/disk/scratch/s1687314/Planning/xsum/jsons/
BERT_DATA_PATH=/disk/scratch/s1687314/Planning/xsum/data/
LOG_PATH=/disk/scratch/s1687314/Planning/xsum/logs/

mkdir ${BERT_DATA_PATH}
rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
	-tokenizer facebook/bart-base \
	-n_cpus 1 \
	-log_file ${LOG_PATH}/preprocess.log
