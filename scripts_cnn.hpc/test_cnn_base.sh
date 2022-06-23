#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.base/
LOG_PATH=${BASE_DIR}/logs.base/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_240000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
	-block_trigram true \
	-max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 20 \
        -test_max_length 350 \
	-visible_gpus 0 \
