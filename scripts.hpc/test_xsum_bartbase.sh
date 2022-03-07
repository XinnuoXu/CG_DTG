#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/models.xsum.bartbase/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs.xsum.bartbase/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_25000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
	-content_planning_model none \
	-tree_gumbel_softmax_tau 0.7 \
	-use_interval true \
	-block_trigram true \
	-max_pos 512 \
	-batch_size 6000 \
	-select_topn 0.3 \
	-visible_gpus 0 \
