#!/bin/bash

BASE_DIR=./outputs.webnlg/

BERT_DATA_PATH=${BASE_DIR}/data.partial_prompt
MODEL_PATH=${BASE_DIR}/models.partial_prompt/
LOG_PATH=${BASE_DIR}/logs.partial_prompt/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
        -model_name t5-small \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_5000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
	-ext_or_abs abs \
        -inference_mode abs \
        -sentence_embedding predicate \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 6000 \
        -test_min_length 5 \
        -test_max_length 250 \
	-visible_gpus 0 \
        -master_port 10008 \
