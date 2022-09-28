#!/bin/bash

BASE_DIR=./outputs.ama/

RAW_PATH=${BASE_DIR}/cluster2cls/
OUTPUT_PATH=${BASE_DIR}/data_cls/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}
rm -rf ${OUTPUT_PATH}/*

python preprocess.py \
	-mode format_for_classification_training \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -saved_tokenizer_path ${OUTPUT_PATH}/tokenizer.pt \
        -tokenizer bert-base-uncased \
	-n_cpus 32 \
        -max_src_ntokens 256 \
        -max_cluster_num 250 \
	-log_file ${LOG_PATH}/preprocess.log
	
