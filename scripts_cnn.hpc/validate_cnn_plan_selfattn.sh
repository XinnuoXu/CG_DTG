#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.plan.self_attn/
ABS_PATH=${BASE_DIR}/models.bartbase/
LOG_PATH=${BASE_DIR}/logs.plan.self_attn/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs marginal_projective_tree \
        -sentence_embedding maxpool \
        -planning_method self_attn \
        -tree_gumbel_softmax_tau 0.3 \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -max_pos 1024 \
	-batch_size 6000 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
