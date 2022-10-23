#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls_v3_sentiment/
MODEL_PATH=${BASE_DIR}/models.selector_sentiment/
LOG_PATH=${DATA_DIR}/logs.selector_sentiment/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
        -model_name allenai/longformer-base-4096 \
        -tokenizer_path allenai/longformer-base-4096 \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-log_file ${LOG_PATH}/train.log \
        -ext_or_abs cls \
        -cls_type sentiment_cls \
        -max_src_nsent 100000 \
	-train_steps 30000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps 1000 \
	-batch_size 3 \
	-report_every 100 \
	-lr 1e-5 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-master_port 10001 \
	-visible_gpus 0,1,2,3
