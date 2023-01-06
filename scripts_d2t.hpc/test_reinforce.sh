#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

MODEL_PATH=${BASE_PATH}/model.test/
DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
LOG_PATH=${BASE_PATH}/logs.re.init_determ/

#MODEL_PATH=${BASE_PATH}/model.re.from_scratch/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.from_scratch/
# 40000

#MODEL_PATH=${BASE_PATH}/model.re.init_determ/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.init_determ/
# 50000

#MODEL_PATH=${BASE_PATH}/model.re.gold_random/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.gold_random/
# 45000

#MODEL_PATH=${BASE_PATH}/model.re.evenly_mix/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.evenly_mix/
# 45000

#MODEL_PATH=${BASE_PATH}/model.re.joint/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.joint/
# 10000

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_2500.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
	-test_alignment_type spectral \
	-test_given_nclusters False \
	-calculate_graph_prob_method min \
	-test_graph_selection_threshold 0.2 \
	-shuffle_src False \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 10 \
        -test_max_length 150 \
	-visible_gpus 0 \

	#-do_analysis True \
