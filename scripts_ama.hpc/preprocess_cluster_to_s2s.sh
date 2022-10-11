#!/bin/bash

BASE_DIR=./outputs.ama/

RAW_PATH=${BASE_DIR}/hdbscan_output/
#RAW_PATH=${BASE_DIR}/augment_output/
OUTPUT_PATH=${BASE_DIR}/seq2seq/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}

python preprocess.py \
	-mode format_hdbscan_cluster_to_s2s \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
	-log_file ${LOG_PATH}/preprocess.log
