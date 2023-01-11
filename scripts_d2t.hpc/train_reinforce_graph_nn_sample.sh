#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

PREVIOUS_MODEL_PATH=${BASE_PATH}/model.re.nn/
MODEL_PATH=${BASE_PATH}/model.re.nn.spectral_with_sample.big_lr/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
#DATA_PATH=${BASE_PATH}/data.test/
LOG_PATH=${BASE_PATH}/logs.re.nn.spectral_with_sample/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-train_from ${PREVIOUS_MODEL_PATH}/model_step_10000.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-nn_graph True \
	-spectral_with_sample True \
	-gold_random_ratio 0.2 \
	-spectral_ratio 0.4 \
	-train_steps 50000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps_reinforce 45000 \
	-warmup_steps 2000 \
	-lr 5e-5 \
	-batch_size 3 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

#-shuffle_src True \
#-load_from_abs ${ABS_MODEL_PATH} \
#ABS_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt
