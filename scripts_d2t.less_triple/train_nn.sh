#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

ENCDEC_PATH=${BASE_PATH}/short.model.re.encdec_partial/
MODEL_PATH=${BASE_PATH}/short.model.re.nn/
DATA_PATH=${BASE_PATH}/short.data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/short.logs.re.nn/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-model_name t5-small \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-train_from ${ENCDEC_PATH}/model_step_5000.pt \
	-seen_predicate_tokenized_paths ${DATA_PATH}/predicate_tokens.txt \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-pretrain_nn_cls True \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-nn_cls_add_negative_samples True \
	-reset_optimizer True \
	-train_steps 12000 \
	-save_checkpoint_steps 2000 \
	-warmup_steps 2000 \
	-lr 1e-2 \
	-batch_size 3000 \
	-report_every 30 \
	-max_pos 250 \
	-max_tgt_len 250 \
        -decay_method linear_warmup \
	-accum_count 1 \
	-visible_gpus 0

