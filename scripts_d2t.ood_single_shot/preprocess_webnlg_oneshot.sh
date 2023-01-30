#!/bin/bash

###############################################
# Shard split
###############################################

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_data.oneshot/
JSON_PATH=${BASE_PATH}/jsons.oneshot/
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
JSON_PATH=${BASE_PATH}/jsons.oneshot/
DATA_PATH=${BASE_PATH}/oneshot.data.base/
LOG_PATH=${BASE_PATH}/logs.data/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
rm -rf ${DATA_PATH}/

python preprocess.py \
	-mode format_for_d2t_base \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-n_cpus 32 \
	-tokenizer t5-small \
	-tokenize_src_predicate True \
        -max_src_ntokens 1024 \
        -max_tgt_ntokens 250 \
	-log_file ${LOG_PATH}/preprocess.log
	
