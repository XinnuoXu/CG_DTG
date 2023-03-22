#!/bin/bash

percent=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/model.re.nn/
LOG_PATH=${BASE_PATH}/logs.re.nn/

# percent=0.005; test_from=500; test_graph_selection_threshold=0.2
# percent=0.01; test_from=500; test_graph_selection_threshold=0.3
# percent=0.05; test_from=500; test_graph_selection_threshold=0.25
# percent=0.1; test_from=1000; test_graph_selection_threshold=0.1

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen False \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder False \
	-test_alignment_type spectral \
	-test_given_nclusters False \
	-test_entity_link True \
	-test_no_single_pred_score True \
	-calculate_graph_prob_method min \
	-test_graph_selection_threshold $3 \
	-nn_graph True \
	-shuffle_src False \
	-max_pos 250 \
	-batch_size 3000 \
        -test_max_length 150 \
        -test_min_length 5 \
	-beam_size 5 \
	-visible_gpus 0 \
        -master_port 10003 \

