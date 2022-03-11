#!/bin/bash

#BASE_DIR=/home/s1687314/Planning/Tree_enc_dec/outputs
BASE_DIR=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/

BERT_DATA_PATH=${BASE_DIR}/data/ 
MODEL_PATH=${BASE_DIR}/models/
LOG_PATH=${BASE_DIR}/logs/

mkdir ${MODEL_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-mode train \
	-ext_or_abs ext \
	-content_planning_model tree \
	-log_file ${LOG_PATH}/train_ext.log \
	-train_steps 150000 \
	-save_checkpoint_steps 30000 \
	-warmup_steps 8000 \
	-batch_size 3000 \
	-report_every 50 \
	-max_pos 1024 \
	-max_tgt_len 250 \
        -ext_dropout 0.1 \
	-lr 0.01 \
	-accum_count 1 \
	-visible_gpus 0

