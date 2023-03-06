#!/bin/bash

ntriple=$1
train_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
PREVIOUS_MODEL_PATH=${BASE_PATH}/short_single.model.re.nn/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.thresholdbase/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.thresholdbase/

mkdir -p ${MODEL_PATH}
rm -rf ${MODEL_PATH}/*
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
        -reinforce_threshold_baseline True \
        -reinforce_baseline_adja_threshold 0.5 \
	-gold_random_ratio 0.1 \
	-spectral_ratio 0.9 \
        -nn_graph_dropout 0.1 \
	-reinforce_bernoulli_temp 1.5 \
	-reset_optimizer True \
	-train_steps 8000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 1000 \
	-batch_size 3 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 3e-6 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 5 \
	-visible_gpus 0
