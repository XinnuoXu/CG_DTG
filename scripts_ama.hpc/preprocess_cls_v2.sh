#!/bin/bash

BASE_DIR=./outputs.ama/

RAW_PATH=${BASE_DIR}/cluster2cls/
OUTPUT_PATH=${BASE_DIR}/data_cls_v2/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}
rm -rf ${OUTPUT_PATH}/*

python preprocess.py \
	-mode format_for_classification_training_v2 \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -saved_tokenizer_path ${OUTPUT_PATH}/tokenizer.pt \
        -tokenizer cardiffnlp/twitter-roberta-base-sentiment \
	-n_cpus 32 \
        -max_src_ntokens 256 \
        -max_cluster_num 1000 \
	-log_file ${LOG_PATH}/preprocess.log
	
        #-tokenizer bert-base-uncased \
