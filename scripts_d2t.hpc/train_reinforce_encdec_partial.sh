#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-pretrain_encoder_decoder True \
	-conditional_decoder True \
	-shuffle_src True \
	-ext_or_abs reinforce \
	-train_steps 4000 \
	-warmup_steps 1000 \
	-save_checkpoint_steps 1000 \
	-lr 3e-4 \
	-batch_size 500 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

#ABS_MODEL_PATH=${BASE_PATH}/model.base/model_step_3000.pt
#-load_from_abs ${ABS_MODEL_PATH} \
#-shuffle_src True \
