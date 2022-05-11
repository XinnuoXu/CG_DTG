#!/bin/bash

BASE_DIR=./outputs.cnn_dm/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.plan.self_attn/
ABS_PATH=${BASE_DIR}/models.bartbase/
LOG_PATH=${BASE_DIR}/logs.plan.self_attn/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
        -ext_or_abs marginal_projective_tree \
        -sentence_embedding maxpool \
        -tree_gumbel_softmax_tau 0.3 \
	-train_steps 320000 \
	-save_checkpoint_steps 40000 \
	-warmup_steps 1000 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-ext_dropout 0.1 \
	-lr 3e-5 \
        -planning_method self_attn \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2
        
#ABS_PATH=${BASE_DIR}/models.bartbase/
        #-load_from_abs ${ABS_PATH}/model_step_320000.pt \
	#-train_from ${MODEL_PATH}/model_step_200000.pt \
