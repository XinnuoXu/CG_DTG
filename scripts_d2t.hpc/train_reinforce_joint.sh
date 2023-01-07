#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/webnlg_data.manual_align/train.jsonl
ENCDEC_PATH=${BASE_PATH}/model.re.encdec_partial/
MODEL_PATH=${BASE_PATH}/model.re.joint/
DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
LOG_PATH=${BASE_PATH}/logs.re.joint/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-deterministic_graph_path ${DETERMINISTIC_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-train_from ${ENCDEC_PATH}/model_step_2000.pt \
	-log_file ${LOG_PATH}/train.log \
	-conditional_decoder True \
	-init_graph_with_deterministic True \
	-gold_random_ratio 0.75 \
	-ext_or_abs reinforce \
	-train_steps 50000 \
	-warmup_steps_reinforce 45000 \
	-lr_encdec 1e-5 \
	-warmup_steps_encdec 25000 \
	-lr_planner 3e-3 \
	-warmup_steps_planner 2000 \
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
