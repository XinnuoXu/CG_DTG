#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

MODEL_PATH=${BASE_PATH}/model.re.base/
DATA_PATH=${BASE_PATH}/data.re.graph_gt/
LOG_PATH=${BASE_PATH}/logs.re.base/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-train_from ${MODEL_PATH}/model_step_2000.pt \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-ext_or_abs reinforce \
	-train_steps 60000 \
	-warmup_steps_reinforce 55000 \
	-warmup_steps 2000 \
	-save_checkpoint_steps 5000 \
	-lr 3e-2 \
	-batch_size 3 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0

#-shuffle_src True \
#-load_from_abs ${ABS_MODEL_PATH} \
#ABS_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt
