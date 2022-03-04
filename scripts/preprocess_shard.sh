#!/bin/bash

#RAW_PATH=../Plan_while_Generate/Content_Selection/cnn_origin_greedy_sent.oracle/
#JSON_PATH=/scratch/xxu/Plan_while_Generate/TreeSumAbs/cnn_dm/jsons/
RAW_PATH=/disk/scratch/s1687314/Planning/xsum/xsum_origin_greedy_sent.oracle/
JSON_PATH=/disk/scratch/s1687314/Planning/xsum/jsons/
LOG_PATH=/disk/scratch/s1687314/Planning/xsum/logs/

mkdir ${JSON_PATH}
rm -rf ${JSON_PATH}/*

python preprocess.py \
	-mode split_shard \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-oracle_topn 1000 \
	-n_cpus 30 \
	-log_file ${LOG_PATH}/preprocess_shard.log \
