#!/bin/bash

BASE_DIR=/home/s1687314/Planning/Tree_enc_dec/outputs

BERT_DATA_PATH=${BASE_DIR}/data/ 
MODEL_PATH=${BASE_DIR}/models/
LOG_PATH=${BASE_DIR}/logs/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_30000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau 0.2 \
	-block_trigram true \
	-max_pos 512 \
	-batch_size 6000 \
	-select_topn 0.3 \
	-visible_gpus 0 \
