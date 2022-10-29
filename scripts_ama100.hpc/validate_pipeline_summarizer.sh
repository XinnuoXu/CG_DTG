#!/bin/bash

BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama100/
DATA_DIR=./outputs.ama100/

DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.summarizer/
LOG_PATH=${BASE_DIR}/logs.summarizer/

python train.py \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -tokenizer_path facebook/bart-base \
	-ext_or_abs abs \
        -max_pos 1024 \
	-batch_size 6000 \
	-max_tgt_len 350 \
	-visible_gpus 0 \
