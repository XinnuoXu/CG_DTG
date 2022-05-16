#!/bin/bash

#BASE_DIR=/disk/scratch/s1687314/Planning/webnlg/extractive.model/
BASE_DIR=./outputs.d2t/

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.ext/
LOG_PATH=${BASE_DIR}/logs.ext/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs ext \
	-content_planning_model tree \
	-result_path ${LOG_PATH}/validation_ext.res \
	-log_file ${LOG_PATH}/validation_ext.log \
	-batch_size 6000 \
	-max_pos 1024 \
	-visible_gpus 0

