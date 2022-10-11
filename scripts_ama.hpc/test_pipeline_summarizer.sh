#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${DATA_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.summarizer.augment/
LOG_PATH=${DATA_DIR}/logs.summarizer/
#LOG_PATH=${BASE_DIR}/logs.summarizer.longer/

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -test_from ${MODEL_PATH}/model_step_300000.pt \
        -tokenizer_path facebook/bart-base \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
        -block_trigram true \
        -max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 3 \
        -test_max_length 250 \
	-visible_gpus 0 \
