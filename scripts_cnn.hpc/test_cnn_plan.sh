#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.plan/
LOG_PATH=${BASE_DIR}/logs.plan/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_280000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
        -prompt_style plan_only \
	-block_trigram true \
        -block_repeat_tok true \
	-max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 20 \
        -test_max_length 150 \
	-visible_gpus 0 \
