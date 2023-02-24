#!/bin/bash

percent=$1
train_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
ENCDEC_PATH=${BASE_PATH}/model.re.encdec_partial.blarge/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/model.re.nn/
LOG_PATH=${BASE_PATH}/logs.re.nn/

# ntriple=0.005; test_from=500
# ntriple=0.01; test_from=1500
# ntriple=0.05; test_from=2500
# ntriple=0.1; test_from=4000

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name facebook/bart-large \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-train_from ${ENCDEC_PATH}/model_step_${train_from}.pt \
	-seen_predicate_tokenized_paths ${DATA_PATH}/predicate_tokens.txt \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-pretrain_nn_cls True \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-nn_cls_add_negative_samples 3 \
	-reset_optimizer True \
	-train_steps 6000 \
	-save_checkpoint_steps 500 \
	-warmup_steps 100 \
	-batch_size 150 \
	-lr 2e-3 \
	-report_every 30 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

	# 0.005
	#-train_steps 500 \
	#-save_checkpoint_steps 100 \
	#-warmup_steps 100 \
	#-batch_size 5 \
	#-lr 1e-3 \

	# 0.01
	#-train_steps 2000 \
	#-save_checkpoint_steps 200 \
	#-warmup_steps 200 \
	#-batch_size 5 \
	#-lr 1e-3 \

	# 0.05
	#-train_steps 6000 \
	#-save_checkpoint_steps 500 \
	#-warmup_steps 100 \
	#-batch_size 50 \
	#-lr 2e-3 \

	#-train_steps 6000 \
	#-save_checkpoint_steps 500 \
	#-warmup_steps 100 \
	#-batch_size 150 \
	#-lr 2e-3 \

