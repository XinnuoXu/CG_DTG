#!/bin/bash

BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/
#BASE_DIR=./outputs.ama/

RAW_PATH=${BASE_DIR}/cluster2cls/
OUTPUT_PATH=${BASE_DIR}/data_cls_v3_noempty/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${OUTPUT_PATH}
rm -r ${OUTPUT_PATH}/*

python preprocess.py \
	-mode format_for_classification_training_v3 \
	-raw_path ${RAW_PATH} \
	-save_path ${OUTPUT_PATH} \
        -saved_tokenizer_path ${OUTPUT_PATH}/tokenizer.pt \
        -tokenizer allenai/longformer-base-4096 \
        -amasum_delete_empty_exampl True \
	-n_cpus 32 \
        -max_src_ntokens 4090 \
	-log_file ${LOG_PATH}/preprocess.log
	
        #-tokenizer bert-base-uncased \
