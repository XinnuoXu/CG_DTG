#!/bin/bash

BASE_DIR=./outputs.ama/

DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.summarizer/
LOG_PATH=${BASE_DIR}/logs.summarizer/

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -test_from ${MODEL_PATH}/model_step_120000.pt \
        -tokenizer_path facebook/bart-base \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
        -block_trigram true \
        -max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 10 \
        -test_max_length 250 \
	-visible_gpus 0 \
