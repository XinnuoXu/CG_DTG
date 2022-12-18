#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
#DATA_PATH=${BASE_PATH}/data.base/
#MODEL_PATH=${BASE_PATH}/model.base/
#LOG_PATH=${BASE_PATH}/logs.base/
DATA_PATH=${BASE_PATH}/test.data/
MODEL_PATH=${BASE_PATH}/model.re/
LOG_PATH=${BASE_PATH}/test.logs/

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
	-train_steps 500 \
	-save_checkpoint_steps 500 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-lr 3e-4 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0

	#-train_steps 8000 \
	#-save_checkpoint_steps 1000 \
