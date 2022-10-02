#!/bin/bash

BASE_DIR=./outputs.ama/
#FILE_NAME=train.0.json
FILE_NAME=$1

RAW_PATH=${BASE_DIR}/hdbscan_output/${FILE_NAME}
OUTPUT_PATH=${BASE_DIR}/augment_output/${FILE_NAME}
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}

python preprocess.py \
	-mode format_augment \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
	-log_file ${LOG_PATH}/preprocess.log
