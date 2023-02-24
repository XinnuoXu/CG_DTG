#!/bin/bash

ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.${tokenizer}/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.${tokenizer}/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name ${tokenizer} \
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
	-batch_size 500 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-lr 5e-4 \
	-label_smoothing 0.0 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

