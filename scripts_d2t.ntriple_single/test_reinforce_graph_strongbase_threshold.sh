#!/bin/bash

ntriple=$1
test_from=$2
test_unseen=$3
selection_threshold=$4

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.strongbase/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.strongbase_threshold/

if [ "$test_unseen" = false ]; then
	OUTPUT_FILE=${LOG_PATH}/test.res
else
	OUTPUT_FILE=${LOG_PATH}/test_unseen.res
fi

mkdir -p ${LOG_PATH}

# [New] strongbaseline training; no sample but with threshold
# ntriple=2; test_from=2000; test_graph_selection_threshold=
# ntriple=3; test_from=4000; test_graph_selection_threshold=
# ntriple=4; test_from=4000; test_graph_selection_threshold=
# ntriple=5; test_from=4000; test_graph_selection_threshold=
# ntriple=6; test_from=6000; test_graph_selection_threshold=
# ntriple=7; test_from=2000; test_graph_selection_threshold=

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen ${test_unseen} \
	-result_path ${OUTPUT_FILE} \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
	-test_alignment_type spectral \
	-test_given_nclusters False \
	-test_entity_link True \
	-test_no_single_pred_score True \
	-calculate_graph_prob_method min \
        -test_run_bernoulli False \
        -test_adja_threshold ${selection_threshold} \
	-test_graph_selection_threshold ${selection_threshold} \
	-nn_graph True \
	-shuffle_src False \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_max_length 150 \
        -test_min_length 5 \
	-beam_size 3 \
	-visible_gpus 0 \
        -reinforce_strong_baseline True \
