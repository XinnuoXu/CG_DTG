#!/bin/bash

BASE_DIR=./outputs.webnlg/

#BERT_DATA_PATH=${BASE_DIR}/data/
#MODEL_PATH=${BASE_DIR}/models.self.discrete/
#LOG_PATH=${BASE_DIR}/logs.self.discrete/
        #-sentence_embedding predicate_meanpool \

BERT_DATA_PATH=${BASE_DIR}/data.pred/
MODEL_PATH=${BASE_DIR}/models.pred.self.discrete/
LOG_PATH=${BASE_DIR}/logs.pred.self.discrete/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_4000.pt \
        -model_name t5-small \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs marginal_projective_tree \
        -planning_method self_attn \
        -sentence_embedding predicate \
        -tree_gumbel_softmax_tau 0.3 \
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
