#!/bin/bash

BASE_DIR=./outputs.ama/
#FILE_NAME=test.0.json
FILE_NAME=$1

RAW_PATH=${BASE_DIR}/raw_split/${FILE_NAME}
OUTPUT_PATH=${BASE_DIR}/cleaned_testset.min10/${FILE_NAME}
#OUTPUT_PATH=${BASE_DIR}/cleaned_testset.min3/${FILE_NAME}
#OUTPUT_PATH=${BASE_DIR}/cleaned_testset/${FILE_NAME}
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}

python preprocess.py \
	-mode format_semantic_cleaning \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
	-log_file ${LOG_PATH}/preprocess.log
