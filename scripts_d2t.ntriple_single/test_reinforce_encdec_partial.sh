#!/bin/bash

ntriple=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
MODEL_PATH=${BASE_PATH}/short_single.model.re.encdec_partial/
DATA_PATH=${BASE_PATH}/short_single.data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/short_single.logs.re.encdec_partial/

# ntriple=2; test_from=1000
# ntriple=3; test_from=2000
# ntriple=4; test_from=2000
# ntriple=5; test_from=3000
# ntriple=6; test_from=3000
# ntriple=7; test_from=3000

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen True \
	-result_path ${LOG_PATH}/test_unseen.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-nn_graph True \
	-conditional_decoder True \
	-test_alignment_type full_src \
	-test_given_nclusters True \
	-shuffle_src False \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 0 \

	#-test_unseen False \
	#-result_path ${LOG_PATH}/test.res \
