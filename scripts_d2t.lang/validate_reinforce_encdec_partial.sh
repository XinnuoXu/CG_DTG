#!/bin/bash

lang=$1 #[br, cy, ga, ru]

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-model_name "facebook/m2m100_418M" \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-ext_or_abs reinforce \
	-pretrain_encoder_decoder True \
	-train_predicate_graph_only False \
	-conditional_decoder True \
	-nn_graph True \
	-shuffle_src False \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
