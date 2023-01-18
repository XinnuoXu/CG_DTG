#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

#MODEL_PATH=${BASE_PATH}/model.re.base/
#DATA_PATH=${BASE_PATH}/data.re.graph_gt/
#LOG_PATH=${BASE_PATH}/logs.re.base/

DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/webnlg_data.manual_align/train.jsonl
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.numerical/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.discriministic/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
	-deterministic_graph_path ${DETERMINISTIC_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_2000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
	-test_alignment_type discriministic \
	-test_given_nclusters False \
	-shuffle_src False \
	-test_graph_selection_threshold 20 \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 10 \
        -test_max_length 150 \
	-visible_gpus 0 \

	#-do_analysis True \
	#-test_graph_selection_threshold $1 \
