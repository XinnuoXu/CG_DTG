#!/bin/bash

BASE_DIR=./outputs.webnlg/

RAW_PATH=../Plan_while_Generate/D2T_data/
ADD_TOKEN_PATH=${RAW_PATH}/webnlg_data/predicates.txt
JSON_PATH=${BASE_DIR}/jsons/
LOG_PATH=${BASE_DIR}/logs/
#BERT_DATA_PATH=${BASE_DIR}/data/
#BERT_DATA_PATH=${BASE_DIR}/data.src_prompt/
#BERT_DATA_PATH=${BASE_DIR}/data.tgt_prompt/
BERT_DATA_PATH=${BASE_DIR}/data.step_wise/
#BERT_DATA_PATH=${BASE_DIR}/data.tgt_intersec/
#BERT_DATA_PATH=${BASE_DIR}/data.soft_src_prompt/

mkdir -p ${LOG_PATH}
mkdir -p ${BERT_DATA_PATH}
rm -rf ${BERT_DATA_PATH}/*

python preprocess.py \
	-mode format_for_training \
        -tokenizer t5-small \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
        -for_stepwise True \
	-n_cpus 32 \
	-log_file ${LOG_PATH}/preprocess.log

        #-add_plan_to_src soft_prompt \
        #-add_plan_to_src hard_prompt \
        #-add_plan_to_tgt prompt \
        #-add_plan_to_tgt intersec \

