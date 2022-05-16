#!/bin/bash

BASE_DIR=./outputs.webnlg/
#BASE_DIR=${SCRATCH_DIR}

BERT_DATA_PATH=${BASE_DIR}/data/
MODEL_PATH=${BASE_DIR}/models.base.test.no_special_token.longtrain/
LOG_PATH=${BASE_DIR}/logs.base.test.no_special_token.longtrain/

#BERT_DATA_PATH=${BASE_DIR}/data.pred/
#MODEL_PATH=${BASE_DIR}/models.pred.base/
#LOG_PATH=${BASE_DIR}/logs.pred.base/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
	-ext_or_abs abs \
	-model_name t5-small \
	-train_steps 12000 \
	-save_checkpoint_steps 4000 \
	-warmup_steps 100 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-ext_dropout 0.1 \
	-lr 3e-4 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2 \
        -master_port 10008 \

