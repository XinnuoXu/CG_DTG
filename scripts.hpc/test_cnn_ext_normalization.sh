#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.ext_normalization//
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/logs.ext_normalization/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_40000.pt \
	-ext_or_abs ext \
	-content_planning_model tree \
	-result_path ${LOG_PATH}/test_ext.res \
	-log_file ${LOG_PATH}/test_ext.log \
	-batch_size 6000 \
	-max_pos 1024 \
        -select_topn 3 \
	-ext_layers 3 \
	-visible_gpus 0 \
	-do_analysis True \

