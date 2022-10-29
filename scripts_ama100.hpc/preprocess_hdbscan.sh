#!/bin/bash

BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama100/ 
#FILE_NAME=train.0.json
FILE_NAME=$1

RAW_PATH=${BASE_DIR}/raw_split/${FILE_NAME}
OUTPUT_PATH=${BASE_DIR}/hdbscan_output/${FILE_NAME}
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}

python preprocess.py \
	-mode format_hdbscan \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -additional_token_path ../Plan_while_Generate/AmaSum/AmaSum_data/common_review.txt \
	-log_file ${LOG_PATH}/preprocess.log
