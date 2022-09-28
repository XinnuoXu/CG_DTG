#!/bin/bash

BASE_DIR=./outputs.ama/

RAW_PATH=${BASE_DIR}/seq2seq/
OUTPUT_PATH=${BASE_DIR}/data/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}
#rm -rf ${OUTPUT_PATH}/*

python preprocess.py \
        -dataset train \
	-mode format_for_training \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -saved_tokenizer_path ${OUTPUT_PATH}/tokenizer.pt \
        -tokenizer facebook/bart-base \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log
	
python preprocess.py \
        -dataset validation \
	-mode format_for_training \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -saved_tokenizer_path ${OUTPUT_PATH}/tokenizer.pt \
        -tokenizer facebook/bart-base \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log
	

RAW_PATH=${BASE_DIR}/seq2seq_test/

python preprocess.py \
        -dataset test \
	-mode format_for_training \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -saved_tokenizer_path ${OUTPUT_PATH}/tokenizer.pt \
        -tokenizer facebook/bart-base \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log
	
