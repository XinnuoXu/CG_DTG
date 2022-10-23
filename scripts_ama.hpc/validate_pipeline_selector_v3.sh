#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls_v3/
MODEL_PATH=${BASE_DIR}/models.selector_v3/
LOG_PATH=${DATA_DIR}/logs.selector_v3/

python train.py  \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
        -tokenizer_path allenai/longformer-base-4096 \
        -model_name allenai/longformer-base-4096 \
        -ext_or_abs cls \
        -cls_type version_3 \
	-batch_size 50 \
	-master_port 10001 \
	-visible_gpus 0
