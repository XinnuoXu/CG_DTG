#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

MODEL_PATH=${BASE_PATH}/model.re.base/
DATA_PATH=${BASE_PATH}/data.re.graph_gt/
LOG_PATH=${BASE_PATH}/logs.re.base/

#MODEL_PATH=${BASE_PATH}/model.re.pure_merge/
#DATA_PATH=${BASE_PATH}/data.re.graph_gt/
#LOG_PATH=${BASE_PATH}/logs.re.pure_merge/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_35000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
	-pretrain_encoder_decoder False \
	-train_predicate_graph_only True \
	-shuffle_src False \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 10 \
        -test_max_length 150 \
	-visible_gpus 0 \

	#-do_analysis True \
