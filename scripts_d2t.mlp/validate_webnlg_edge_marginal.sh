#!/bin/bash

BASE_DIR=./outputs.webnlg/
#BASE_DIR=${SCRATCH_DIR}

#BERT_DATA_PATH=${BASE_DIR}/data/
#MODEL_PATH=${BASE_DIR}/models.edge.marginal/
#LOG_PATH=${BASE_DIR}/logs.edge.marginal/
        #-sentence_embedding predicate_meanpool \

BERT_DATA_PATH=${BASE_DIR}/data.pred/
MODEL_PATH=${BASE_DIR}/models.pred.edge.marginal/
LOG_PATH=${BASE_DIR}/logs.pred.edge.marginal/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-model_name t5-small \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-ext_or_abs marginal_projective_tree \
        -sentence_embedding predicate \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
        -predicates_start_from_id 32101 \
        -tree_gumbel_softmax_tau -1 \
	-log_file ${LOG_PATH}/validation.log \
        -result_path ${LOG_PATH}/validation.res \
	-batch_size 6000 \
	-max_pos 250 \
	-max_tgt_len 250 \
	-visible_gpus 0

