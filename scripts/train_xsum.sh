#!/bin/bash

#BASE_DIR=/home/s1687314/Planning/Tree_enc_dec/outputs
BASE_DIR=/scratch/xxu/Plan_while_Generate/TreeSumAbs/xsum/

BERT_DATA_PATH=${BASE_DIR}/data/ 
MODEL_PATH=${BASE_DIR}/models/
LOG_PATH=${BASE_DIR}/logs/

mkdir ${MODEL_PATH}

python train.py  \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-mode train \
	-ext_or_abs abs \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau 0.2 \
        -freeze_encoder_decoder True \
	-log_file ${LOG_PATH}/train.log \
	-train_steps 60000 \
	-save_checkpoint_steps 5000 \
	-warmup_steps 500 \
	-batch_size 3000 \
	-report_every 50 \
	-max_pos 1024 \
	-max_tgt_len 250 \
	-lr_tmt 1e-2 \
	-lr_enc_dec 3e-5 \
	-decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2

