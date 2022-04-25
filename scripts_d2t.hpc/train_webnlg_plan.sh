#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/models.plan/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/logs.plan/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_name t5-base \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-mode train \
	-ext_or_abs plan \
	-content_planning_model tree \
	-log_file ${LOG_PATH}/train.log \
	-train_steps 12000 \
	-save_checkpoint_steps 4000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-ext_dropout 0.1 \
	-lr 3e-4 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0
