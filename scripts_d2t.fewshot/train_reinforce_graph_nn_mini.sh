#!/bin/bash

percent=$1
train_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
PREVIOUS_MODEL_PATH=${BASE_PATH}/model.re.nn/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/model.re.nn.spectral/
LOG_PATH=${BASE_PATH}/logs.re.nn.spectral/

# percent=0.005; train_from=400;
# percent=0.01; train_from=1200;
# percent=0.05; train_from=3500;
# percent=0.1; train_from=5000;

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name facebook/bart-large \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-train_from ${PREVIOUS_MODEL_PATH}/model_step_${train_from}.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-nn_graph True \
	-gold_random_ratio 0.4 \
	-spectral_ratio 0.1 \
	-reset_optimizer True \
	-train_steps 2000 \
	-save_checkpoint_steps 200 \
	-warmup_steps_reinforce 1900 \
	-warmup_steps 100 \
	-batch_size 5 \
	-lr 1e-6 \
	-report_every 10 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0


	#-train_steps 5000 \
	#-save_checkpoint_steps 500 \
	#-warmup_steps_reinforce 4500 \
	#-warmup_steps 2000 \
	#-batch_size 5 \
	#-lr 3e-6 \
