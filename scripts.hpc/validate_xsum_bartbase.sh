#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/models.xsum.bartbase/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs.xsum.bartbase/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-ext_or_abs abs \
	-content_planning_model none \
	-tree_gumbel_softmax_tau 0.7 \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-test_all \
        -max_pos 1024 \
	-batch_size 6000 \
	-visible_gpus 0 \
