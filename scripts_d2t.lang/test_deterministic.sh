#!/bin/bash

lang=$1 #[br, cy, ga, ru]
test_from=$2
selection_threshold=$3

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/multiL_${lang}/webnlg_data.manual_align/train.jsonl
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.discriministic/

# ntriple=br; test_from=5000; 1
# ntriple=cy; test_from=5000; 1
# ntriple=ga; test_from=8000; 1
# ntriple=ru; test_from=7000; 4


mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
	-deterministic_graph_path ${DETERMINISTIC_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen False \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-test_lang ${lang} \
	-nn_graph True \
	-conditional_decoder False \
	-test_alignment_type discriministic \
	-test_given_nclusters False \
	-test_graph_selection_threshold ${selection_threshold} \
	-shuffle_src False \
        -test_no_repeat_ngram_size 4 \
	-max_pos 250 \
	-batch_size 3 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \
	-master_port 10005 \
