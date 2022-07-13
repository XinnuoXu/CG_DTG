#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data.parallel/
MODEL_PATH=${BASE_DIR}/models.parallel/
LOG_PATH=${BASE_DIR}/logs.parallel/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-ext_or_abs abs \
        -prompt_style tgt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -max_pos 1024 \
	-batch_size 6000 \
	-max_pos 1024 \
	-max_tgt_len 500 \
	-visible_gpus 0 \
