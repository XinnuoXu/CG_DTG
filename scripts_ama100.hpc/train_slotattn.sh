#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.ama100/
DATA_PATH=${BASE_PATH}/test.data/
MODEL_PATH=${BASE_PATH}/models.kmeans/
LOG_PATH=${BASE_PATH}/logs/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name facebook/bart-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs slotsumm \
	-cluster_algorithm 'soft_kmeans' \
	-slotsumm_train_stage 'slot_attn' \
	-train_from ${MODEL_PATH}/model_step_12000.pt \
	-train_steps 20000 \
	-lr_planner 3e-4 \
	-lr_encdec 3e-3 \
	-warmup_steps_planner 1000 \
	-warmup_steps_encdec 1000 \
	-save_checkpoint_steps 1500 \
	-batch_size 30 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 800 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0

	#-slotsumm_train_stage 'pre-train' \
	#-lr_encdec 3e-4 \

	#-slotsumm_train_stage 'gold_align' \
	#-train_from ${MODEL_PATH}/model_step_4500.pt \

	#-cluster_algorithm 'slot_attn' \
