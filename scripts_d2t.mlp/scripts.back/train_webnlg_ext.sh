#!/bin/bash

BASE_DIR=/disk/scratch/s1687314/Planning/webnlg/extractive.model/

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.ext/
LOG_PATH=${BASE_DIR}/logs.ext/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -model_name ./outputs.cnn_dm/model.pt \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-mode train \
	-ext_or_abs ext \
	-content_planning_model tree \
	-log_file ${LOG_PATH}/train.log \
	-train_steps 20000 \
	-save_checkpoint_steps 4000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 150 \
	-max_tgt_len 150 \
	-ext_dropout 0.1 \
	-lr 0.01 \
	-accum_count 1 \
	-ext_layers 3 \
	-visible_gpus 0
