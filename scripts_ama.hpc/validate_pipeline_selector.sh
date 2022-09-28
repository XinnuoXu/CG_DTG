#!/bin/bash

BASE_DIR=./outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls/
MODEL_PATH=${BASE_DIR}/models.selector/
LOG_PATH=${BASE_DIR}/logs.selector/

python train.py  \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -tokenizer_path bert-base-uncased \
        -model_name bert-base-uncased \
        -ext_or_abs cls \
	-batch_size 6000 \
	-master_port 10001 \
	-visible_gpus 0
