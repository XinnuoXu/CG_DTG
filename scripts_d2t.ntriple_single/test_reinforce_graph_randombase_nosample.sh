#!/bin/bash

ntriple=$1
test_from=$2
test_unseen=$3
selection_threshold=$4

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
MODEL_PATH=${BASE_PATH}/short_single.model.re.nn.randombase/
LOG_PATH=${BASE_PATH}/short_single.logs.re.nn.randombase_nosample/

if [ "$test_unseen" = false ]; then
	OUTPUT_FILE=${LOG_PATH}/test.res
else
	OUTPUT_FILE=${LOG_PATH}/test_unseen.res
fi

mkdir -p ${LOG_PATH}

#-gold_random_ratio 0.4 -spectral_ratio 0.1 [CHOOSE]
# ntriple=2; test_from=1000; test_graph_selection_threshold=0.06 --> 44.62
# ntriple=3; test_from=1000; test_graph_selection_threshold=0.7 --> 52.54
# ntriple=4; test_from=4500; test_graph_selection_threshold=0.56 --> 55.19
# ntriple=5; test_from=3000; test_graph_selection_threshold=0.5 --> 58.49
# ntriple=6; test_from=1000; test_graph_selection_threshold=0.38 --> 58.31
# ntriple=7; test_from=3000; test_graph_selection_threshold=0.52 --> 57.41

# [New] no sample in inference
# ntriple=2; test_from=6000; test_graph_selection_threshold=0.5
# ntriple=3; test_from=5000; test_graph_selection_threshold=0.54
# ntriple=4; test_from=8000; test_graph_selection_threshold=0.76
# ntriple=5; test_from=4000; test_graph_selection_threshold=0.52
# ntriple=6; test_from=7000; test_graph_selection_threshold=0.56
# ntriple=7; test_from=8000; test_graph_selection_threshold=0.58

# [Ori] no sample in inference
# ntriple=2; test_from=8000; test_graph_selection_threshold=0.54
# ntriple=3; test_from=7000; test_graph_selection_threshold=0.02
# ntriple=4; test_from=8000; test_graph_selection_threshold=0.34
# ntriple=5; test_from=4000; test_graph_selection_threshold=0.48
# ntriple=6; test_from=7000; test_graph_selection_threshold=0.42
# ntriple=7; test_from=2000; test_graph_selection_threshold=0.50

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
        -nn_graph_dropout 0.0 \
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
        -seed 666 \
        -reinforce_strong_baseline False \
