#!/bin/bash

BASE_DIR=./outputs.webnlg/

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.base/
LOG_PATH=${BASE_DIR}/logs.base/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
        -model_name t5-small \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_3000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
        -inference_mode abs \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 6000 \
        -test_min_length 10 \
        -test_max_length 250 \
	-visible_gpus 0 \
