#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.ext/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/logs.ext/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-mode train \
	-ext_or_abs ext \
	-content_planning_model tree \
        -freeze_encoder_decoder True \
	-log_file ${LOG_PATH}/train_ext.log \
	-train_steps 80000 \
	-save_checkpoint_steps 20000 \
	-warmup_steps 3000 \
	-batch_size 3000 \
	-report_every 50 \
	-max_pos 1024 \
	-max_tgt_len 250 \
        -ext_dropout 0.1 \
	-lr 0.01 \
	-accum_count 2 \
	-visible_gpus 0,1,2

