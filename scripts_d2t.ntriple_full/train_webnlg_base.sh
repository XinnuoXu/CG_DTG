#!/bin/bash

tokenizer=$1 #[t5-small, t5-base, t5-large]

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/baselines/
DATA_PATH=${BASE_PATH}/data.${tokenizer}/
MODEL_PATH=${BASE_PATH}/model.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.${tokenizer}/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}
rm ${MODEL_PATH}/*

python train.py  \
	-mode train \
	-model_name ${tokenizer} \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs abs \
	-train_steps 100000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps 1000 \
	-batch_size 8 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 3e-5 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0,1,2

