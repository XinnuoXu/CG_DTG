#!/bin/bash

BASE_DIR=./outputs.webnlg/

BERT_DATA_PATH=${BASE_DIR}/data.plan/
MODEL_PATH=${BASE_DIR}/models.plan_generation/
LOG_PATH=${BASE_DIR}/logs.plan_generation/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs abs \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-max_pos 250 \
	-batch_size 6000 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
