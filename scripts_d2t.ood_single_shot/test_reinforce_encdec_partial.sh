#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

MODEL_PATH=${BASE_PATH}/oneshot.model.re.encdec_partial.numerical/
DATA_PATH=${BASE_PATH}/data.oneshot/
LOG_PATH=${BASE_PATH}/oneshot.logs.re.encdec_partial_numerical/

#MODEL_PATH=${BASE_PATH}/oneshot.model.re.encdec_partial/
#DATA_PATH=${BASE_PATH}/data.oneshot/
#LOG_PATH=${BASE_PATH}/oneshot.logs.re.encdec_partial/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_100.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
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

	#-nn_graph True \
