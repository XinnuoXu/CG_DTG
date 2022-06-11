#!/bin/bash

BASE_DIR=./outputs.webnlg/

#BERT_DATA_PATH=${BASE_DIR}/data.tgt_prompt/
#MODEL_PATH=${BASE_DIR}/models.tgt_prompt/
#LOG_PATH=${BASE_DIR}/logs.tgt_prompt/

BERT_DATA_PATH=${BASE_DIR}/data.single_sentences_tgt_prompts/
MODEL_PATH=${BASE_DIR}/models.tgt_prompt_parallel/
LOG_PATH=${BASE_DIR}/logs.tgt_prompt_parallel/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
        -model_name t5-small \
	-input_path ${BERT_DATA_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_6000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
        -pred_special_tok '<PRED>' \
        -obj_special_tok '<OBJ>' \
	-ext_or_abs abs \
        -inference_mode tgt_prompt \
        -sentence_embedding predicate \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 6000 \
        -test_min_length 10 \
        -test_max_length 250 \
	-visible_gpus 0 \
        -do_analysis \
