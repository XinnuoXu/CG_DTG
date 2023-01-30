#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
DATA_PATH=${BASE_PATH}/data.parallel/
MODEL_PATH=${BASE_PATH}/models.parallel/
LOG_PATH=${BASE_PATH}/logs.parallel/

mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_name t5-base \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-ext_or_abs abs \
        -max_pos 1024 \
	-batch_size 6000 \
	-max_tgt_len 150 \
	-visible_gpus 0 \
