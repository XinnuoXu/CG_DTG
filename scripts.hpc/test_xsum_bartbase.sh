#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/models.xsum.bartbase/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs/logs.xsum.bartbase/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_90000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs abs \
	-content_planning_model none \
	-use_interval true \
	-block_trigram true \
	-max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 10 \
        -test_max_length 60 \
	-visible_gpus 0 \
