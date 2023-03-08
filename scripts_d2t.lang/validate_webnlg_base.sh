#!/bin/bash

tokenizer=$1 #[t5-small, t5-base, t5-large]

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/baselines/
DATA_PATH=${BASE_PATH}/data.${tokenizer}/
MODEL_PATH=${BASE_PATH}/model.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.${tokenizer}/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-ext_or_abs abs \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
