#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

ENCDEC_PATH=${BASE_PATH}/model.re.encdec_partial.numerical/
MODEL_PATH=${BASE_PATH}/model.re.from_scratch/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
#DATA_PATH=${BASE_PATH}/data.test/
LOG_PATH=${BASE_PATH}/logs.re.from_scratch/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-train_from ${ENCDEC_PATH}/model_step_7000.pt \
	-log_file ${LOG_PATH}/train.log \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-ext_or_abs reinforce \
	-gold_random_ratio 0.75 \
	-train_steps 20000 \
	-save_checkpoint_steps 2000 \
	-warmup_steps_reinforce 19000 \
	-warmup_steps 7000 \
	-lr 8e-2 \
	-batch_size 3 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

