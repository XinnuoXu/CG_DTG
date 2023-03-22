#!/bin/bash

ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]
test_from=$3
test_unseen=$4

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.${tokenizer}/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.${tokenizer}/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.${tokenizer}/

if [ "$test_unseen" = false ]; then
	OUTPUT_FILE=${LOG_PATH}/test.res
else
	OUTPUT_FILE=${LOG_PATH}/test_unseen.res
fi

# t5-base; ntriple=2; test_from=10000
# t5-base; ntriple=3; test_from=10000
# t5-base; ntriple=4; test_from=15000
# t5-base; ntriple=7; test_from=30000

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
	-nn_graph True \
	-conditional_decoder False \
	-test_alignment_type full_src \
	-test_given_nclusters True \
	-shuffle_src False \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \
