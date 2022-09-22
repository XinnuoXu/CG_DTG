#!/bin/bash

BASE_DIR=./outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls/
MODEL_PATH=${BASE_DIR}/models.selector/
LOG_PATH=${BASE_DIR}/logs.selector/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
        -model_name bert-base-uncased \
        -tokenizer_path bert-base-uncased \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-log_file ${LOG_PATH}/train.log \
        -ext_or_abs cls \
        -max_src_nsent 5000 \
        -ext_ff_size 1024 \
        -ext_heads 8 \
	-ext_dropout 0.1 \
        -ext_layers 6 \
	-train_steps 150000 \
	-save_checkpoint_steps 30000 \
	-warmup_steps 1000 \
	-batch_size 100 \
	-report_every 100 \
	-lr 3e-5 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-master_port 10001 \
	-visible_gpus 0,1,2,3
