#!/bin/bash

ntriple=$1
train_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
PREVIOUS_MODEL_PATH=${BASE_PATH}/short_single.model.re.nn/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.randombase/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.randombase/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-train_from ${PREVIOUS_MODEL_PATH}/model_step_${train_from}.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-nn_graph True \
	-gold_random_ratio 0.2 \
	-spectral_ratio 0.8 \
	-reset_optimizer True \
	-train_steps 8000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps_reinforce 7000 \
	-warmup_steps 2000 \
	-batch_size 3 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 3e-6 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 5 \
	-visible_gpus 0
