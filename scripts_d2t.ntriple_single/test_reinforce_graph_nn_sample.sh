#!/bin/bash

ntriple=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.spectral_with_sample/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.spectral_with_sample/

# ntriple=2; test_from=3500; test_graph_selection_threshold=0.98
# ntriple=3; test_from=4500; test_graph_selection_threshold=0.98(?)
# ntriple=4; test_from=3500; test_graph_selection_threshold=0.78
# ntriple=5; test_from=4500; test_graph_selection_threshold=0.58
# ntriple=6; test_from=2500; test_graph_selection_threshold=0.34
# ntriple=7; test_from=3500; test_graph_selection_threshold=0.54

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
	-conditional_decoder True \
	-test_alignment_type spectral \
	-test_given_nclusters False \
	-test_entity_link True \
	-test_no_single_pred_score True \
	-calculate_graph_prob_method min \
	-test_graph_selection_threshold $3 \
	-nn_graph True \
	-shuffle_src False \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_max_length 150 \
        -test_min_length 5 \
	-beam_size 3 \
	-visible_gpus 0 \

	#-test_unseen True \
	#-result_path ${LOG_PATH}/test_unseen.res \
