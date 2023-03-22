#!/bin/bash

ntriple=$1
train_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
ENCDEC_PATH=${BASE_PATH}/short_single.model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn/

# ntriple=2; test_from=2000
# ntriple=3; test_from=3000
# ntriple=4; test_from=4000
# ntriple=5; test_from=5000
# ntriple=6; test_from=6000
# ntriple=7; test_from=8000

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}
rm ${MODEL_PATH}/*

python train.py  \
	-mode train \
	-model_name t5-base \
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
	-lr 1e-3 \
	-report_every 30 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

	#-nn_cls_add_negative_samples 0 \
