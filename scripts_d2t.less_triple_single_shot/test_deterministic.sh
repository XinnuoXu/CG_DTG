#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

DETERMINISTIC_PATH=../Plan_while_Generate/D2T_data/webnlg.3triple_oneshot/webnlg_data.manual_align/train.jsonl
MODEL_PATH=${BASE_PATH}/short_single.model.re.encdec_partial.numerical/
#DATA_PATH=${BASE_PATH}/short_single.data.re.merge.tokenized_preds/
DATA_PATH=${BASE_PATH}/short_single.data.re.align.tokenized_preds/
LOG_PATH=${BASE_PATH}/short_single.logs.re.discriministic/

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
	-deterministic_graph_path ${DETERMINISTIC_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_2000.pt \
	-test_unseen False \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs reinforce \
	-conditional_decoder True \
	-test_alignment_type discriministic \
	-test_given_nclusters False \
	-shuffle_src False \
	-test_graph_selection_threshold 2 \
	-block_trigram true \
	-max_pos 250 \
	-batch_size 3000 \
        -test_min_length 5 \
        -test_max_length 150 \
	-beam_size 3 \
	-visible_gpus 2 \

	#-test_unseen True \
	#-result_path ${LOG_PATH}/test_unseen.res \
	#-test_graph_selection_threshold $1 \