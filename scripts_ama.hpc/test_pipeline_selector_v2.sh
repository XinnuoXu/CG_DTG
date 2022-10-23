#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls_v2/
MODEL_PATH=${BASE_DIR}/models.selector_v2/
LOG_PATH=${DATA_DIR}/logs.selector_v2/

python train.py  \
	-mode test \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -test_from ${MODEL_PATH}/model_step_15000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
        -tokenizer_path cardiffnlp/twitter-roberta-base-sentiment \
        -model_name cardiffnlp/twitter-roberta-base-sentiment \
        -ext_or_abs cls \
        -cls_type version_2 \
	-batch_size 6000 \
	-master_port 10001 \
	-visible_gpus 0
