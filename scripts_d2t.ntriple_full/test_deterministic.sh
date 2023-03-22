#!/bin/bash

ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]
test_from=$3
test_unseen=$4
selection_threshold=$5

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/webnlg.${ntriple}triple_full/webnlg_data.manual_align/train.jsonl
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.${tokenizer}/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.re.discriministic.${tokenizer}/

if [ "$test_unseen" = false ]; then
        OUTPUT_FILE=${LOG_PATH}/test.res
else
        OUTPUT_FILE=${LOG_PATH}/test_unseen.res
fi

# ntriple=2; test_from=10000; test_graph_selection_threshold=1
# ntriple=3; test_from=10000; test_graph_selection_threshold=1
# ntriple=4; test_from=15000; test_graph_selection_threshold=1
# ntriple=7; test_from=30000; test_graph_selection_threshold=1

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
	-conditional_decoder True \
	-test_alignment_type discriministic \
	-test_given_nclusters False \
	-shuffle_src False \
        -nn_graph True \
	-test_graph_selection_threshold ${selection_threshold} \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \

