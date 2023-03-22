#!/bin/bash

percent=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/fewshot_webnlg_percent_${percent}/fewshot_webnlg.manual_align/train.jsonl
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds//
LOG_PATH=${BASE_PATH}/logs.re.discriministic/

# ntriple=0.005; test_from=500; test_graph_selection_threshold=2
# ntriple=0.01; test_from=500; test_graph_selection_threshold=1
# ntriple=0.05; test_from=2500; test_graph_selection_threshold=2
# ntriple=0.1; test_from=4500; test_graph_selection_threshold=1

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-model_name facebook/bart-base \
	-input_path ${DATA_PATH} \
	-deterministic_graph_path ${DETERMINISTIC_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen False \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
        -nn_graph True \
	-conditional_decoder False \
	-test_alignment_type discriministic \
	-test_given_nclusters False \
	-shuffle_src False \
	-test_graph_selection_threshold $3 \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 5 \
        -seed 42 \
	-visible_gpus 0 \
