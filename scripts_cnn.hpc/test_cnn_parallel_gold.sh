#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data.parallel.gold/
MODEL_PATH=${BASE_DIR}/models.parallel/
LOG_PATH=${BASE_DIR}/logs.parallel.gold/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_320000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
        -prompt_style tgt \
        -inference_mode tgt_prompt \
        -downstream_task summarization \
	-block_trigram true \
	-max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 20 \
        -test_max_length 500 \
	-visible_gpus 0 \
