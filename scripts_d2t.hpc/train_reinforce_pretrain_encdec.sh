#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
ABS_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt
MODEL_PATH=${BASE_PATH}/model.re/
DATA_PATH=${BASE_PATH}/data.re/
LOG_PATH=${BASE_PATH}/logs.re/

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
	-train_steps 6000 \
	-warmup_steps 1000 \
	-save_checkpoint_steps 1000 \
	-lr 3e-4 \
	-batch_size 2 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

