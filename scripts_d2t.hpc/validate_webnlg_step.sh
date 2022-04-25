#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/models.step/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/logs.step/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-model_name t5-base \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs step \
	-content_planning_model none \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -max_pos 1024 \
	-batch_size 6000 \
	-max_pos 150 \
	-max_tgt_len 150 \
	-visible_gpus 0 \
