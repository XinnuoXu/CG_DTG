#!/bin/bash

# Setup for Webnlg
BASE_DIR=./outputs.webnlg/

RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_sentences/
JSON_PATH=${BASE_DIR}/jsons.sentences/
LOG_PATH=${BASE_DIR}/logs/

mkdir -p ${LOG_PATH}
mkdir -p ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
        -mode split_shard \
        -dataset train \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log \

python preprocess.py \
        -mode split_shard \
        -dataset validation \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log \


RAW_PATH=../Plan_while_Generate/D2T_data/webnlg_data/
JSON_PATH=${BASE_DIR}/jsons.sentences/
LOG_PATH=${BASE_DIR}/logs/

PRED_PLAN_PATH=${BASE_DIR}/logs.plan_generation/test.res.4000.candidate
PRED_PLAN_ID_PATH=${BASE_DIR}/logs.plan_generation/test.res.4000.eid

python preprocess.py \
        -mode split_shard_with_predicted_plan_parallel \
        -dataset test \
        -raw_path ${RAW_PATH} \
        -save_path ${JSON_PATH} \
        -predicted_plan_path ${PRED_PLAN_PATH} \
        -predicted_plan_id_path ${PRED_PLAN_ID_PATH} \
        -oracle_topn 1000 \
        -n_cpus 30 \
        -log_file ${LOG_PATH}/preprocess_shard.log \
