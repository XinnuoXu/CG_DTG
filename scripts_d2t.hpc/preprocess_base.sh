#!/bin/bash

# Shard split
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
#RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_data/
#JSON_PATH=${BASE_PATH}/jsons/
#LOG_PATH=${BASE_PATH}/logs/
RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_test/
JSON_PATH=${BASE_PATH}/test.jsons/
LOG_PATH=${BASE_PATH}/test.logs/

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



# Prepare for base encoder-decoder training
ADD_TOKEN_PATH=../Plan_while_Generate/D2T_data/webnlg_data/predicates.txt
#BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
#JSON_PATH=${BASE_PATH}/jsons/
#DATA_PATH=${BASE_PATH}/data.base/
#LOG_PATH=${BASE_PATH}/logs.base/
BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/
JSON_PATH=${BASE_PATH}/test.jsons/
DATA_PATH=${BASE_PATH}/test.data/
LOG_PATH=${BASE_PATH}/test.logs/

mkdir -p ${LOG_PATH}
mkdir -p ${DATA_PATH}
rm -rf ${DATA_PATH}/*

python preprocess.py \
	-mode format_for_d2t_base \
	-raw_path ${JSON_PATH} \
	-save_path ${DATA_PATH} \
	-additional_token_path ${ADD_TOKEN_PATH} \
        -saved_tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-n_cpus 32 \
	-tokenizer t5-base \
        -max_src_ntokens 1024 \
        -max_tgt_ntokens 250 \
	-log_file ${LOG_PATH}/preprocess.log
	
