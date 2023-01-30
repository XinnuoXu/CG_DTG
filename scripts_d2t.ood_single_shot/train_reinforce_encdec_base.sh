#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
TRAIN_FROM_PATH=${BASE_PATH}/model.re.encdec_base/ 
MODEL_PATH=${BASE_PATH}/oneshot.model.re.encdec_base/
DATA_PATH=${BASE_PATH}/data.oneshot/
LOG_PATH=${BASE_PATH}/oneshot.logs.re.base/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-train_from ${TRAIN_FROM_PATH}/model_step_7000.pt \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-shuffle_src True \
	-reset_optimizer True \
	-train_steps 500 \
	-save_checkpoint_steps 50 \
	-warmup_steps 200 \
	-batch_size 10 \
	-report_every 50 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 5e-5 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

