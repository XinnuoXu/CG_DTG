#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls_v2/
MODEL_PATH=${BASE_DIR}/models.selector_v2/
LOG_PATH=${DATA_DIR}/logs.selector_v2/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
        -model_name cardiffnlp/twitter-roberta-base-sentiment \
        -tokenizer_path cardiffnlp/twitter-roberta-base-sentiment \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-log_file ${LOG_PATH}/train.log \
        -ext_or_abs cls \
        -cls_type version_2 \
        -max_src_nsent 10000 \
        -ext_ff_size 2048 \
        -ext_heads 8 \
	-ext_dropout 0.1 \
        -ext_layers 6 \
	-train_steps 30000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps 1000 \
	-batch_size 100 \
	-report_every 100 \
	-lr 1e-5 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-master_port 10001 \
	-visible_gpus 0,1,2,3

        #-tokenizer_path bert-base-uncased \
        #-max_src_nsent 5000 \
