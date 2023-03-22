#!/bin/bash

percent=$1

###############################################
# Shard split
###############################################

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
RAW_PATH=../Plan_while_Generate/D2T_data/fewshot_webnlg_percent_${percent}/fewshot_webnlg.manual_align/
JSON_PATH=${BASE_PATH}/jsons.re.align.rule_based/
LOG_PATH=${BASE_PATH}/logs.data/

mkdir -p ${LOG_PATH}
mkdir -p ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
        -mode split_shard \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log



###############################################
# Setup for sentence-level generation
###############################################

ADD_TOKEN_PATH=../Plan_while_Generate/D2T_data/fewshot_webnlg/predicates.txt
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
JSON_PATH=${BASE_PATH}/jsons.re.align.rule_based/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.data/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
#rm -rf ${DATA_PATH}/*
rm -rf ${DATA_PATH}/test.*

python preprocess.py \
	-mode format_sentence_level \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-seen_predicate_paths ${ADD_TOKEN_PATH} \
	-seen_predicate_tokenized_paths ${DATA_PATH}/predicate_tokens.txt \
	-remove_single_triple_datapoints True \
	-remove_noise_datapoints False \
	-tokenize_src_predicate True \
	-multi_ref_test True \
	-n_cpus 32 \
	-tokenizer facebook/bart-base \
        -max_src_ntokens 1024 \
        -max_tgt_ntokens 250 \
	-log_file ${LOG_PATH}/preprocess.log
