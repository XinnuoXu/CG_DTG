#!/bin/bash

BASE_DIR=./outputs.webnlg/

BERT_DATA_PATH=${BASE_DIR}/data.plan/
MODEL_PATH=${BASE_DIR}/models.plan_generation/
LOG_PATH=${BASE_DIR}/logs.plan_generation/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
        -model_name t5-small \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_2000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
	-ext_or_abs abs \
        -inference_mode plan \
        -sentence_embedding predicate \
	-block_trigram true \
	-max_pos 20 \
	-batch_size 6000 \
        -test_min_length 0 \
        -test_max_length 20 \
        -beam_size 5 \
	-visible_gpus 0 \
