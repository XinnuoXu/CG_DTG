#!/bin/bash

BASE_DIR=./outputs.webnlg/
#BASE_DIR=${SCRATCH_DIR}

ABS_PATH=${BASE_DIR}/models.base/

BERT_DATA_PATH=${BASE_DIR}/data.single_sentences_step_wise/
MODEL_PATH=${BASE_DIR}/models.step_wise_parallel
LOG_PATH=${BASE_DIR}/logs.step_wise_parallel

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -load_from_abs ${ABS_PATH}/model_step_3000.pt \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
        -log_gradient ${LOG_PATH}/gradient.log \
	-ext_or_abs step \
        -cross_attn_weight_format hard \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
        -predicates_start_from_id 32101 \
	-model_name t5-small \
        -tree_info_dim 512 \
        -ext_ff_size 1024 \
	-train_steps 8000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 500 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-ext_dropout 0.1 \
	-lr 3e-5 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2 \
        -master_port 10006 \
