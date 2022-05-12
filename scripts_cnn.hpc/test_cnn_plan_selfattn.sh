#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.plan.self_attn/
LOG_PATH=${BASE_DIR}/logs.plan.self_attn/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_200000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs marginal_projective_tree \
        -sentence_embedding maxpool \
        -tree_gumbel_softmax_tau 0.3 \
	-block_trigram true \
	-max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 55 \
        -test_max_length 140 \
	-visible_gpus 0 \
