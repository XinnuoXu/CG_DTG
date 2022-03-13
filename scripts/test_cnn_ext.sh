#!/bin/bash

BASE_DIR=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/

BERT_DATA_PATH=${BASE_DIR}/data/ 
MODEL_PATH=${BASE_DIR}/models/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_150000.pt \
	-ext_or_abs ext \
	-content_planning_model tree \
	-result_path ${LOG_PATH}/test_ext.res \
	-log_file ${LOG_PATH}/test_ext.log \
	-batch_size 6000 \
	-max_pos 1024 \
        -select_topn 3 \
	-ext_layers 3 \
	-visible_gpus 1

