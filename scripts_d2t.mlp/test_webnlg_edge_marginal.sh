#!/bin/bash

BASE_DIR=./outputs.webnlg/

#BERT_DATA_PATH=${BASE_DIR}/data/
#MODEL_PATH=${BASE_DIR}/models.edge.marginal/
#LOG_PATH=${BASE_DIR}/logs.edge.marginal/
        #-sentence_embedding predicate_meanpool \

BERT_DATA_PATH=${BASE_DIR}/data.pred/
MODEL_PATH=${BASE_DIR}/models.pred.edge.marginal/
LOG_PATH=${BASE_DIR}/logs.pred.edge.marginal/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_4000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
        -model_name t5-small \
	-ext_or_abs marginal_projective_tree \
        -sentence_embedding predicate \
        -tree_gumbel_softmax_tau -1 \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
	-inference_mode abs \
        -predicates_start_from_id 32101 \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 6000 \
        -test_min_length 10 \
        -test_max_length 250 \
	-visible_gpus 0 \
