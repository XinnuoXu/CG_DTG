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
	-conditional_decoder True \
	-ext_or_abs reinforce \
	-train_steps 60000 \
	-warmup_steps_reinforce 55000 \
	-lr_encdec 3e-5 \
	-warmup_steps_encdec 10000 \
	-lr_planner 3e-2 \
	-warmup_steps_planner 1 \
	-save_checkpoint_steps 5000 \
	-batch_size 3 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

#-shuffle_src True \
#-train_predicate_graph_only True \
#-lr 3e-2 \
#-warmup_steps 2000 \
