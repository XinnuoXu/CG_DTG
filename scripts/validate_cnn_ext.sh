#!/bin/bash

#BASE_DIR=/home/s1687314/Planning/Tree_enc_dec/outputs
BASE_DIR=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/

BERT_DATA_PATH=${BASE_DIR}/data/ 
MODEL_PATH=${BASE_DIR}/models/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-ext_or_abs ext \
	-content_planning_model tree \
	-result_path ${LOG_PATH}/validation_ext.res \
	-log_file ${LOG_PATH}/validation_ext.log \
	-batch_size 6000 \
	-max_pos 1024 \
	-ext_layers 3 \
	-visible_gpus 0

