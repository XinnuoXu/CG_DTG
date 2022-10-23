#!/bin/bash

DATA_DIR=./outputs.ama/
BASE_DIR=/rds/user/hpcxu1/hpc-work/outputs.ama/

DATA_PATH=${BASE_DIR}/data_cls_v3/
MODEL_PATH=${BASE_DIR}/models.selector_select/
LOG_PATH=${DATA_DIR}/logs.selector_select/

python train.py  \
	-mode test \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -test_from ${MODEL_PATH}/model_step_30000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
        -tokenizer_path allenai/longformer-base-4096 \
        -model_name allenai/longformer-base-4096 \
        -ext_or_abs cls \
        -cls_type select \
	-batch_size 10 \
	-master_port 10001 \
	-visible_gpus 0
