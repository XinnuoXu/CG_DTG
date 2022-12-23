#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
MODEL_PATH=${BASE_PATH}/model.re.strong_reward/
DATA_PATH=${BASE_PATH}/data.re/
LOG_PATH=${BASE_PATH}/logs.re.strong_reward/
#MODEL_PATH=${BASE_PATH}/model.re/
#DATA_PATH=${BASE_PATH}/data.re/
#LOG_PATH=${BASE_PATH}/logs.re/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
        -max_pos 1024 \
	-batch_size 3 \
	-max_tgt_len 250 \
	-visible_gpus 0 \

	#-pretrain_encoder_decoder True \
