#!/bin/bash

BASE_DIR=./outputs.webnlg/
#BASE_DIR=${SCRATCH_DIR}

#BERT_DATA_PATH=${BASE_DIR}/data/
#MODEL_PATH=${BASE_DIR}/models.edge.discrete/
#LOG_PATH=${BASE_DIR}/logs.edge.discrete/
        #-sentence_embedding predicate_meanpool \

BERT_DATA_PATH=${BASE_DIR}/data.pred/
MODEL_PATH=${BASE_DIR}/models.pred.edge.discrete/
LOG_PATH=${BASE_DIR}/logs.pred.edge.discrete/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode train \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-log_file ${LOG_PATH}/train.log \
        -log_gradient ${LOG_PATH}/gradient.log \
	-ext_or_abs marginal_projective_tree \
        -sentence_embedding predicate \
        -tree_gumbel_softmax_tau 0.3 \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
        -predicates_start_from_id 32101 \
	-model_name t5-small \
        -tree_info_dim 512 \
        -ext_ff_size 1024 \
	-train_steps 6000 \
	-save_checkpoint_steps 1000 \
	-warmup_steps 500 \
	-batch_size 3000 \
	-report_every 100 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-ext_dropout 0.1 \
	-lr 3e-4 \
        -decay_method linear_warmup \
	-accum_count 2 \
	-visible_gpus 0,1,2 \
        -master_port 10000 \
