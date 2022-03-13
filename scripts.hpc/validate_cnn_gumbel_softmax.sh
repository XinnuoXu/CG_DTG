#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.gumbel_softmax_3/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/logs.gumbel_softmax_3/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-ext_or_abs abs \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau 0.5 \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -max_pos 1024 \
	-batch_size 6000 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
