#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
MODEL_PATH=${BASE_PATH}/short.model.re.encdec_base/
DATA_PATH=${BASE_PATH}/short.data.re.base/
LOG_PATH=${BASE_PATH}/short.logs.re.base/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-shuffle_src True \
	-train_steps 10000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 1000 \
	-batch_size 500 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 5e-4 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0,1,2

