#!/bin/bash

BASE_DIR=/home/s1687314/Planning/Tree_enc_dec/outputs

BERT_DATA_PATH=${BASE_DIR}/data/ 
MODEL_PATH=${BASE_DIR}/models/
LOG_PATH=${BASE_DIR}/logs/

mkdir ${MODEL_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-mode train \
	-ext_or_abs abs \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau 0.2 \
        -freeze_encoder_decoder True \
	-log_file ${LOG_PATH}/train.log \
	-train_steps 60000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps 10000 \
	-batch_size 10 \
	-report_every 50 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-use_interval true \
	-lr 5e-3 \
	-accum_count 5 \
	-visible_gpus 0

