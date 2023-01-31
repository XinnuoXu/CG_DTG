#!/bin/bash

###############################################
# Shard split
###############################################

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
RAW_PATH=../Plan_while_Generate/D2T_data/webnlg.3triple_oneshot/webnlg_data.manual_align/
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

ADD_TOKEN_PATH=../Plan_while_Generate/D2T_data/webnlg_data/predicates.txt
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
JSON_PATH=${BASE_PATH}/jsons.re.align.rule_based/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
#DATA_PATH=${BASE_PATH}/data.test/
LOG_PATH=${BASE_PATH}/logs.data/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
rm -rf ${DATA_PATH}/test.*

python preprocess.py \
	-mode format_sentence_level \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-seen_predicate_tokenized_paths ${DATA_PATH}/predicate_tokens.txt \
	-remove_single_triple_datapoints True \
	-remove_noise_datapoints False \
	-tokenize_src_predicate True \
	-multi_ref_test True \
	-n_cpus 32 \
	-tokenizer t5-small \
        -max_src_ntokens 1024 \
        -max_tgt_ntokens 250 \
	-log_file ${LOG_PATH}/preprocess.log
