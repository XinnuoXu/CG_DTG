#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
ABS_MODEL_PATH=${BASE_PATH}/model.re/model_step_500.pt
MODEL_PATH=${BASE_PATH}/model.re/
DATA_PATH=${BASE_PATH}/test.data/
LOG_PATH=${BASE_PATH}/test.logs/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-load_from_abs ${ABS_MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-ext_or_abs reinforce \
	-train_steps 600 \
	-save_checkpoint_steps 600 \
	-warmup_steps 100 \
	-lr 3e-4 \
	-batch_size 3000 \
	-report_every 10 \
	-max_pos 1024 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0

