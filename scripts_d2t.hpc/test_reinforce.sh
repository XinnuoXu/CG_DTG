#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
#MODEL_PATH=${BASE_PATH}/model.re.strong_reward/
#DATA_PATH=${BASE_PATH}/data.re/
#LOG_PATH=${BASE_PATH}/logs.re.strong_reward/
MODEL_PATH=${BASE_PATH}/model.re/
DATA_PATH=${BASE_PATH}/data.re/
LOG_PATH=${BASE_PATH}/logs.re/


mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_6000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-block_trigram true \
	-max_pos 1024 \
	-batch_size 2 \
        -test_min_length 5 \
        -test_max_length 150 \
	-visible_gpus 0 \

	#-do_analysis True \
