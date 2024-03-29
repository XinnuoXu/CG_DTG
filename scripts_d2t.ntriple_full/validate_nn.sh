#!/bin/bash

ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds.${tokenizer}/
MODEL_PATH=${BASE_PATH}/model.re.nn.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.re.nn.${tokenizer}/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-pretrain_nn_cls True \
	-pretrain_encoder_decoder False \
	-train_predicate_graph_only True \
	-conditional_decoder False \
	-shuffle_src False \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
