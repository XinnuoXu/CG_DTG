#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
DATA_PATH=${BASE_PATH}/test.data/
MODEL_PATH=${BASE_PATH}/test.models.marginal.kmeans/
LOG_PATH=${BASE_PATH}/test.logs.marginal.kmeans/
#PRETRAINED_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt
PRETRAINED_MODEL_PATH=${MODEL_PATH}/model_step_500.pt

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
        -model_name t5-base \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_600.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-load_pretrained_model ${PRETRAINED_MODEL_PATH} \
	-ext_or_abs slot \
	-slot_sample_mode 'marginal' \
	-block_trigram true \
	-max_pos 150 \
	-batch_size 6000 \
        -test_min_length 0 \
        -test_max_length 150 \
	-visible_gpus 0 \

	#-cluster_algorithm 'soft_kmeans' \

