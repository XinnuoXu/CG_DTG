#!/bin/bash

ntriple=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/

DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn/

# ntriple=2; test_from=2000; test_graph_selection_threshold=0.92/0.34 --> 44.51/44.49
# ntriple=3; test_from=2000; test_graph_selection_threshold=0.76/0.26 --> 51.97/52.65
# ntriple=4; test_from=3000; test_graph_selection_threshold=0.7/0.24 --> 55.36/55.32
# ntriple=5; test_from=4000; test_graph_selection_threshold=0.54/0.23 --> 58.49/57.59
# ntriple=6; test_from=2000; test_graph_selection_threshold=0.56/0.21 --> 58.35/57.94
# ntriple=7; test_from=3000; test_graph_selection_threshold=0.5/0.24 --> 57.49/57.80

# v2: new negtive sample, do it on all examples [CHOOSE]
# ntriple=2; test_from=2000; test_graph_selection_threshold=0.26 --> 44.64
# ntriple=3; test_from=2000; test_graph_selection_threshold=0.72 --> 52.42
# ntriple=4; test_from=2000; test_graph_selection_threshold=0.56 --> 55.04
# ntriple=5; test_from=3000; test_graph_selection_threshold=0.5 --> 58.40
# ntriple=6; test_from=2000; test_graph_selection_threshold=0.48 --> 58.27
# ntriple=7; test_from=3000; test_graph_selection_threshold=0.54 --> 57.51

# v2.1: new negtive sample, only do it on one-sentence-output examples
# ntriple=2; test_from=; test_graph_selection_threshold= --> 44.78
# ntriple=3; test_from=; test_graph_selection_threshold= --> 52.36
# ntriple=4; test_from=; test_graph_selection_threshold= --> 55.15
# ntriple=5; test_from=; test_graph_selection_threshold= --> 58.08
# ntriple=6; test_from=; test_graph_selection_threshold= --> 58.23
# ntriple=7; test_from=; test_graph_selection_threshold= --> 57.78

# v2.3: new negtive sample, only do it on <=4 triple examples
# ntriple=4; test_from=; test_graph_selection_threshold= --> 55
# ntriple=5; test_from=; test_graph_selection_threshold= --> 58.28
# ntriple=6; test_from=3000; test_graph_selection_threshold= --> 58.24
# ntriple=7; test_from=2000; test_graph_selection_threshold= --> 57.72

# v2.4: new negtive sample, only do it on all example, negtive sample == #input triple
# ntriple=2; test_from=3000; test_graph_selection_threshold= --> 44.80
# ntriple=3; test_from=2000; test_graph_selection_threshold= --> 52.12
# ntriple=4; test_from=3000; test_graph_selection_threshold= --> 55.19
# ntriple=5; test_from=4000; test_graph_selection_threshold= --> 58.03
# ntriple=6; test_from=4000; test_graph_selection_threshold= --> 57.70
# ntriple=7; test_from=3000; test_graph_selection_threshold= --> 57.39

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen True \
	-result_path ${LOG_PATH}/test_unseen.res \
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

	#-test_unseen False \
	#-result_path ${LOG_PATH}/test.res \
