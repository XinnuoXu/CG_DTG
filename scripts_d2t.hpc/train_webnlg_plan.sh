#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
DATA_PATH=${BASE_PATH}/data/
MODEL_PATH=${BASE_PATH}/models.plan/
LOG_PATH=${BASE_PATH}/logs.plan/
#MODEL_PATH=${BASE_PATH}/models.plan.from_raw/
#LOG_PATH=${BASE_PATH}/logs.plan.from_raw/
PRETRAINED_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-load_pretrained_model ${PRETRAINED_MODEL_PATH} \
	-ext_or_abs slot \
	-train_steps 6000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-lr 3e-4 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

