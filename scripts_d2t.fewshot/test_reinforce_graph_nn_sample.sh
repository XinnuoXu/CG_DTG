#!/bin/bash

ntriple=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.spectral_with_sample/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.spectral_with_sample/

#-gold_random_ratio 0.4 \
#-spectral_ratio 0.4 \
#-spectral_with_sample True \
# ntriple=2; test_from=3500; test_graph_selection_threshold=0.98/0.22 --> 44.76
# ntriple=3; test_from=4500; test_graph_selection_threshold=0.98(?)/0.27 --> 51.36
# ntriple=4; test_from=3500; test_graph_selection_threshold=0.78/0.22 --> 55.72
# ntriple=5; test_from=4500; test_graph_selection_threshold=0.58/0.23 --> 58.46
# ntriple=6; test_from=2500; test_graph_selection_threshold=0.34/0.23 --> 58.14
# ntriple=7; test_from=3500; test_graph_selection_threshold=0.54/0.23 --> 57.64

#-gold_random_ratio 0.3 \
#-spectral_ratio 0.2 \
#-spectral_with_sample True \
# ntriple=2; test_from=5000; test_graph_selection_threshold= --> 44.47
# ntriple=3; test_from=4500; test_graph_selection_threshold= --> 52.12
# ntriple=4; test_from=5000; test_graph_selection_threshold= --> 54.90
# ntriple=5; test_from=5000; test_graph_selection_threshold= --> 58.47
# ntriple=6; test_from=2500; test_graph_selection_threshold= --> 58.34
# ntriple=7; test_from=3500; test_graph_selection_threshold= --> 57.57

#-gold_random_ratio 0.4 \
#-spectral_ratio 0.1 \
#-spectral_with_sample True \ [CHOOSE]
# ntriple=2; test_from=500; test_graph_selection_threshold=0.7 --> 44.66
# ntriple=3; test_from=500; test_graph_selection_threshold=0.72 --> 52.54
# ntriple=4; test_from=5000; test_graph_selection_threshold=0.58 --> 55.23
# ntriple=5; test_from=1500; test_graph_selection_threshold=0.5 --> 58.35
# ntriple=6; test_from=5000; test_graph_selection_threshold=0.4 --> 58.2
# ntriple=7; test_from=4000; test_graph_selection_threshold=0.5 --> 57.43

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
