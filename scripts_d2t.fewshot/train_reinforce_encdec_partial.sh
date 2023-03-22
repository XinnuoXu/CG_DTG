#!/bin/bash

percent=$1

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}
#rm ${MODEL_PATH}/*

python train.py  \
	-mode train \
	-model_name facebook/bart-base \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-train_steps 7000 \
	-save_checkpoint_steps 500 \
	-warmup_steps 100 \
	-batch_size 32 \
	-report_every 100 \
	-max_pos 256 \
	-max_tgt_len 256 \
	-lr 3e-5 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

	#-shuffle_src True \
