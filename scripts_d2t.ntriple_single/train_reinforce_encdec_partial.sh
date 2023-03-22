#!/bin/bash

ntriple=$1

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
MODEL_PATH=${BASE_PATH}/short_single.model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/short_single.data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/short_single.logs.re.encdec_partial/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}
rm -rf ${MODEL_PATH}/*

python train.py  \
	-mode train \
	-model_name t5-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-train_steps 12000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 1000 \
	-batch_size 32 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 3e-5 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

	#-shuffle_src True \
