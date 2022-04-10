#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/models.ext/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/logs.base/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_20000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs ext \
        -content_planning_model tree \
	-block_trigram true \
	-max_pos 150 \
	-batch_size 6000 \
        -test_min_length 10 \
        -test_max_length 150 \
	-visible_gpus 0 \
