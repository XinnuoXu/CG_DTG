#!/bin/bash

ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]
train_from=$3

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
PREVIOUS_MODEL_PATH=${BASE_PATH}/model.re.nn.${tokenizer}/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds.${tokenizer}/
MODEL_PATH=${BASE_PATH}/model.re.nn.spectral.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.re.nn.spectral.${tokenizer}/

mkdir -p ${MODEL_PATH}
rm -rf ${MODEL_PATH}/*
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name ${tokenizer} \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
        -train_from ${PREVIOUS_MODEL_PATH}/model_step_${train_from}.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
	-train_predicate_graph_only True \
	-nn_graph True \
	-gold_random_ratio 0.0 \
	-spectral_ratio 1.0 \
        -nn_graph_dropout 0.0 \
	-reinforce_bernoulli_temp 0.8 \
	-reset_optimizer True \
	-train_steps 7000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 1000 \
	-batch_size 300 \
	-report_every 20 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-min_tgt_len 5 \
	-lr 5e-4 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0
	
	#-train_predicate_graph_only True \
