#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
DATA_PATH=${BASE_PATH}/data/
MODEL_PATH=${BASE_PATH}/models.plan/
LOG_PATH=${BASE_PATH}/logs.plan/
PRETRAINED_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode validate \
	-model_name t5-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/validation.log \
        -result_path ${LOG_PATH}/validation.res \
	-load_pretrained_model ${PRETRAINED_MODEL_PATH} \
	-ext_or_abs slot \
	-batch_size 6000 \
	-max_pos 1024 \
	-max_tgt_len 150 \
	-visible_gpus 0
