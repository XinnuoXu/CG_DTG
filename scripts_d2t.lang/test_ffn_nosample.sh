#!/bin/bash

lang=$1 #[br, cy, ga, ru]
test_from=$2
selection_threshold=$3

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/model.re.nn/
LOG_PATH=${BASE_PATH}/logs.re.nn.ffn_nosample/

# ntriple=br; test_from=4500; 0.005
# ntriple=cy; test_from=3000; 0.005
# ntriple=ga; test_from=4500; 0.005
# ntriple=ru; test_from=4000; 0.14

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
	-test_lang ${lang} \
	-shuffle_src False \
	-nn_graph True \
	-conditional_decoder False \
	-test_alignment_type spectral \
	-test_given_nclusters False \
	-test_entity_link True \
	-test_no_single_pred_score True \
	-calculate_graph_prob_method min \
        -test_run_bernoulli False \
        -test_adja_threshold -1 \
	-test_graph_selection_threshold ${selection_threshold} \
        -test_no_repeat_ngram_size 4 \
	-max_pos 250 \
	-batch_size 3 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \
	-master_port 10006 \
