#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${DATA_DIR}/data_cls/
MODEL_PATH=${BASE_DIR}/models.selector/
LOG_PATH=${DATA_DIR}/logs.selector/

python train.py  \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -model_name cardiffnlp/twitter-roberta-base-sentiment \
        -tokenizer_path cardiffnlp/twitter-roberta-base-sentiment \
        -ext_or_abs cls \
        -cls_type version_1 \
	-batch_size 6000 \
	-master_port 10001 \
	-visible_gpus 0
