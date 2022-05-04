#!/bin/bash

BASE_DIR=./outputs.webnlg/

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.plan/
LOG_PATH=${BASE_DIR}/logs.plan/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-model_name t5-small \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs marginal_projective_tree \
	-content_planning_model tree \
        -predicates_start_from_id 32101 \
	-log_file ${LOG_PATH}/validation.log \
        -result_path ${LOG_PATH}/validation.res \
	-batch_size 6000 \
	-max_pos 1024 \
	-max_tgt_len 150 \
        -tree_gumbel_softmax_tau 0.2 \
	-visible_gpus 0
