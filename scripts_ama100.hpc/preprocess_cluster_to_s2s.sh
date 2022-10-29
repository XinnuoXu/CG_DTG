#!/bin/bash

BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama100/

RAW_PATH=${BASE_DIR}/hdbscan_output/
OUTPUT_PATH=${BASE_DIR}/seq2seq/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${OUTPUT_PATH}
mkdir -p ${LOG_PATH}

python preprocess.py \
	-mode format_hdbscan_cluster_to_s2s \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
	-log_file ${LOG_PATH}/preprocess.log
