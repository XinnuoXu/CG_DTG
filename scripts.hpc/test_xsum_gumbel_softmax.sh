#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/models.gumbel_softmax/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs.gumbel_softmax/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_60000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau 0.5 \
	-block_trigram true \
        -max_pos 1024 \
	-batch_size 6000 \
	-visible_gpus 0 \
