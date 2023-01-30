#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
DATA_PATH=${BASE_PATH}/data.parallel/
MODEL_PATH=${BASE_PATH}/models.parallel/
LOG_PATH=${BASE_PATH}/logs.parallel/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-input_path ${DATA_PATH} \
	-model_name t5-base \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs abs \
	-train_steps 20000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-lr 3e-4 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2
