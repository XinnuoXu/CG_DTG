#!/bin/bash

ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]
test_from=$3
test_unseen=$4
selection_threshold=$5

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds.${tokenizer}/
MODEL_PATH=${BASE_PATH}/model.re.nn.spectral.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.re.nn.randombase_nosample.${tokenizer}/

if [ "$test_unseen" = false ]; then
	OUTPUT_FILE=${LOG_PATH}/test.res
else
	OUTPUT_FILE=${LOG_PATH}/test_unseen.res
fi

# ntriple=2; test_from=4000; test_graph_selection_threshold=0.06
# ntriple=3; test_from=4000; test_graph_selection_threshold=0.66
# ntriple=4; test_from=4000; test_graph_selection_threshold=0.48
# ntriple=7; test_from=4000; test_graph_selection_threshold=0.40

# [New] no sample in inference
# ntriple=2; test_from=3000; test_graph_selection_threshold=
# ntriple=3; test_from=4000; test_graph_selection_threshold=
# ntriple=4; test_from=6000; test_graph_selection_threshold=
# ntriple=7; test_from=7000; test_graph_selection_threshold=

mkdir -p ${LOG_PATH}

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
        -test_adja_threshold -1 \
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
        -reinforce_strong_baseline False \
