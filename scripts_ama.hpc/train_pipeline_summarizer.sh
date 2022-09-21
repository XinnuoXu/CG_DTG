#!/bin/bash

BASE_DIR=./outputs.ama/

DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.summarizer/
LOG_PATH=${BASE_DIR}/logs.summarizer/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
        -model_name facebook/bart-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path facebook/bart-base \
	-log_file ${LOG_PATH}/train.log \
        -ext_or_abs abs \
	-train_steps 320000 \
	-save_checkpoint_steps 40000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 350 \
	-ext_dropout 0.1 \
	-lr 3e-5 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0
