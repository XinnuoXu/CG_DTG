#!/bin/bash

BASE_DIR=./outputs.ama/

RAW_PATH=${BASE_DIR}/logs.selector/
OUTPUT_PATH=${BASE_DIR}/seq2seq_test/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}

python preprocess.py \
	-mode format_selected_cluster_to_s2s \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -amasum_verdict_cluster_topk 3 \
        -amasum_pros_cluster_topk 6 \
        -amasum_cons_cluster_topk 3 \
        -amasum_random_topk False \
	-log_file ${LOG_PATH}/preprocess.log
