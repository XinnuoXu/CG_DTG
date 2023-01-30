#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
DATA_PATH=${BASE_PATH}/test.data/
MODEL_PATH=${BASE_PATH}/test.models.marginal.kmeans/
LOG_PATH=${BASE_PATH}/test.logs.marginal.kmeans/
PRETRAINED_MODEL_PATH=${BASE_PATH}/model.test/model_step_1000.pt

mkdir -p ${MODEL_PATH}
rm ${MODEL_PATH}/model_step_600.pt
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
	-slot_sample_mode 'marginal' \
	-lr_planner 1e-2 \
	-lr_encdec 1e-5 \
	-slot_iters 3 \
	-warmup_steps_planner 10 \
	-warmup_steps_encdec 300 \
	-train_steps 600 \
	-save_checkpoint_steps 600 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0

	#-log_gradient ./log_gradient.txt \
	#-cluster_algorithm 'soft_kmeans' \
