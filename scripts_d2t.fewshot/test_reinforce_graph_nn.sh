#!/bin/bash

ntriple=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.spectral/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.spectral/

#-gold_random_ratio 0.7 -spectral_ratio 0.1
# ntriple=2; test_from=3500; test_graph_selection_threshold=0.98/0.29 --> 44.74
# ntriple=3; test_from=5000; test_graph_selection_threshold=0.98(?)/0.27 --> 51.41
# ntriple=4; test_from=4000; test_graph_selection_threshold=0.86/0.25 --> 55.66
# ntriple=5; test_from=4500; test_graph_selection_threshold=0.6/0.2 --> 58.33
# ntriple=6; test_from=2500; test_graph_selection_threshold=0.44/0.23 --> 58.21
# ntriple=7; test_from=4000; test_graph_selection_threshold=0.62/0.23 --> 57.41

#-gold_random_ratio 0.9 -spectral_ratio 0.1
# ntriple=2; test_from=5000; test_graph_selection_threshold=44.46
# ntriple=3; test_from=4500; test_graph_selection_threshold=52.46
# ntriple=4; test_from=4000; test_graph_selection_threshold=55.55
# ntriple=5; test_from=4500; test_graph_selection_threshold=58.3
# ntriple=6; test_from=2000; test_graph_selection_threshold=58.25
# ntriple=7; test_from=3500; test_graph_selection_threshold=57.54

#-gold_random_ratio 0.4 -spectral_ratio 0.1 [CHOOSE]
# ntriple=2; test_from=1000; test_graph_selection_threshold=0.06 --> 44.62
# ntriple=3; test_from=1000; test_graph_selection_threshold=0.7 --> 52.54
# ntriple=4; test_from=4500; test_graph_selection_threshold=0.56 --> 55.19
# ntriple=5; test_from=3000; test_graph_selection_threshold=0.5 --> 58.49
# ntriple=6; test_from=1000; test_graph_selection_threshold=0.38 --> 58.31
# ntriple=7; test_from=3000; test_graph_selection_threshold=0.52 --> 57.41

mkdir -p ${LOG_PATH}
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
	#-test_graph_selection_threshold $3 \
