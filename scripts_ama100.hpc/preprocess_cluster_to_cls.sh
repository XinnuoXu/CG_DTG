#!/bin/bash

BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama100/
#FILE_NAME=train.0.json
FILE_NAME=$1

RAW_PATH=${BASE_DIR}/hdbscan_output/${FILE_NAME}
OUTPUT_PATH=${BASE_DIR}/cluster2cls/${FILE_NAME}
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
#mkdir -p ${OUTPUT_PATH}

python preprocess.py \
	-mode format_hdbscan_cluster_to_cls \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
	-n_cpus 1 \
	-log_file ${LOG_PATH}/preprocess.log
