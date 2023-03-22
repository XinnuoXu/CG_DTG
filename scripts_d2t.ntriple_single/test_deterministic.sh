#!/bin/bash

ntriple=$1
test_from=$2
test_unseen=$3
selection_threshold=$4

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/webnlg.${ntriple}triple_oneshot/webnlg_data.manual_align/train.jsonl
MODEL_PATH=${BASE_PATH}/short_single.model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/short_single.logs.re.discriministic/

if [ "$test_unseen" = false ]; then
	OUTPUT_FILE=${LOG_PATH}/test.res
else
	OUTPUT_FILE=${LOG_PATH}/test_unseen.res
fi

# ntriple=2; test_from=2000; test_graph_selection_threshold=1
# ntriple=3; test_from=3000; test_graph_selection_threshold=1
# ntriple=4; test_from=4000; test_graph_selection_threshold=1
# ntriple=5; test_from=5000; test_graph_selection_threshold=1
# ntriple=6; test_from=6000; test_graph_selection_threshold=1
# ntriple=7; test_from=8000; test_graph_selection_threshold=1

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
	-deterministic_graph_path ${DETERMINISTIC_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen ${test_unseen} \
	-result_path ${OUTPUT_FILE} \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder False \
	-test_alignment_type discriministic \
	-test_given_nclusters False \
	-shuffle_src False \
        -nn_graph True \
	-test_graph_selection_threshold ${selection_threshold} \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \

