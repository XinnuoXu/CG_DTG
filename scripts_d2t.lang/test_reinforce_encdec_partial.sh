#!/bin/bash

lang=$1 #[br, cy, ga, ru]
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial/

# ntriple=br; test_from=5000
# ntriple=cy; test_from=5000
# ntriple=ga; test_from=8000
# ntriple=ru; test_from=7000

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen false \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-test_lang ${lang} \
	-nn_graph True \
	-conditional_decoder False \
	-test_alignment_type full_src \
	-test_given_nclusters True \
	-shuffle_src False \
        -test_no_repeat_ngram_size 4 \
	-max_pos 250 \
	-batch_size 3 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \
	-master_port 10005 \
