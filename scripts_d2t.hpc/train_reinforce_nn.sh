#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

ENCDEC_PATH=${BASE_PATH}/model.re.encdec_partial/
MODEL_PATH=${BASE_PATH}/model.re.nn/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.nn/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-train_from ${ENCDEC_PATH}/model_step_2000.pt \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-nn_graph True \
	-pretrain_nn_cls True \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-ext_or_abs reinforce \
	-train_steps 10000 \
	-warmup_steps 2000 \
	-save_checkpoint_steps 2000 \
	-lr 5e-3 \
	-batch_size 3000 \
	-report_every 30 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0

#-shuffle_src True \
#-load_from_abs ${ABS_MODEL_PATH} \
#ABS_MODEL_PATH=${BASE_PATH}/model.base/model_step_4000.pt