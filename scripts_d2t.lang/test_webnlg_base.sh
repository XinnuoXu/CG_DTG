#!/bin/bash

tokenizer=$1 #[t5-small, t5-base, t5-large]

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/baselines/
DATA_PATH=${BASE_PATH}/data.${tokenizer}/
MODEL_PATH=${BASE_PATH}/model.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.${tokenizer}/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_10000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 6000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \
