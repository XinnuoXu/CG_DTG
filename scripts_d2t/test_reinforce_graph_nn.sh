#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

MODEL_PATH=${BASE_PATH}/model.re.nn.spectral/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.nn.spectral/

#MODEL_PATH=${BASE_PATH}/model.re.nn/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
#LOG_PATH=${BASE_PATH}/logs.re.nn/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_6000.pt \
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
	-test_graph_selection_threshold 0.15 \
	-nn_graph True \
	-shuffle_src False \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_max_length 150 \
        -test_min_length 5 \
	-beam_size 3 \
	-visible_gpus 0 \

	#-test_graph_selection_threshold $1 \
	#-test_unseen False \
	#-result_path ${LOG_PATH}/test.res \
