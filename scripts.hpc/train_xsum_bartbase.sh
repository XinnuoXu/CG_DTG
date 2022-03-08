#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/models.xsum.bartbase/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs.xsum.bartbase/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -train_from ${MODEL_PATH}/model_step_70000.pt \
	-mode train \
	-ext_or_abs abs \
	-content_planning_model none \
	-log_file ${LOG_PATH}/train.log \
	-train_steps 100000 \
	-save_checkpoint_steps 10000 \
	-warmup_steps 20000 \
	-batch_size 3000 \
	-report_every 50 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-use_interval true \
	-ext_dropout 0.1 \
	-lr 2e-3 \
	-accum_count 5 \
	-visible_gpus 0,1,2
