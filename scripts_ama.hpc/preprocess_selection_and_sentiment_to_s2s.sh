#!/bin/bash

BASE_DIR=./outputs.ama/

SELECTION_PATH=${BASE_DIR}/logs.selector_select/
SENTIMENT_PATH=${BASE_DIR}/logs.selector_sentiment/
OUTPUT_PATH=${BASE_DIR}/seq2seq_test/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}

python preprocess.py \
	-mode format_selection_and_sentiment_to_s2s \
	-test_selection_path ${SELECTION_PATH} \
	-test_sentiment_path ${SENTIMENT_PATH} \
	-save_path ${OUTPUT_PATH} \
        -amasum_verdict_cluster_topk 3 \
        -amasum_pros_cluster_topk 6 \
        -amasum_cons_cluster_topk 3 \
        -amasum_verdict_sentiment_threshold 0.0 \
        -amasum_pros_sentiment_threshold 0.0 \
        -amasum_cons_sentiment_threshold 0.0 \
        -amasum_binary_selection_topk 12 \
        -amasum_binary_selection_threshold 0.00 \
        -amasum_random_topk False \
	-log_file ${LOG_PATH}/preprocess.log
