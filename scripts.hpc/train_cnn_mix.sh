#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.gumbel_softmax_3/
EXT_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.ext/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/logs.gumbel_softmax_3/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-mode train \
	-ext_or_abs abs \
        -load_from_ext ${EXT_PATH}/model_step_60000.pt \
        -abs_plus_ext_loss True \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau 0.5 \
	-log_file ${LOG_PATH}/train.log \
	-train_steps 320000 \
	-save_checkpoint_steps 40000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-lr 3e-5 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2
