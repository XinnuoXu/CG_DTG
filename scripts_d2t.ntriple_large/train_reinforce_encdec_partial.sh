#!/bin/bash

ntriple=$1

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
DATA_PATH=${BASE_PATH}/short_large.data.re.merge.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_large.model.re.encdec_partial/
LOG_PATH=${BASE_PATH}/short_large.logs.re.encdec_partial/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-large \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-shuffle_src True \
	-train_steps 10000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 1000 \
	-batch_size 5 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 5e-4 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

