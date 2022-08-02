#!/bin/bash

BASE_DIR=./outputs.webnlg/

BERT_DATA_PATH=${BASE_DIR}/data.prefix_tgt
MODEL_PATH=${BASE_DIR}/models.prefix_tgt/
LOG_PATH=${BASE_DIR}/logs.prefix_tgt/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs abs \
        -prefix_tgt_training True \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-max_pos 250 \
	-batch_size 6000 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
        -master_port 10008 \
