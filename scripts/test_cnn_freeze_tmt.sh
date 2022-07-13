#!/bin/bash

BERT_DATA_PATH=/disk/scratch/s1687314/Planning/cnn/

MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.freeze_tmt/
EXT_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.ext/
ABS_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/models.bartbase/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.cnn_dm/logs.freeze_tmt/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_320000.pt \
        -load_from_ext ${EXT_PATH}/model_step_60000.pt \
        -load_from_abs ${ABS_PATH}/model_step_320000.pt \
	-result_path ${LOG_PATH}/test.res \
	-log_file ${LOG_PATH}/test.log \
	-ext_or_abs mix \
	-content_planning_model tree \
	-tree_gumbel_softmax_tau -1 \
        -abs_plus_ext_loss 0 \
        -ext_topn 3 \
	-block_trigram true \
	-max_pos 1024 \
	-batch_size 6000 \
        -test_min_length 55 \
        -test_max_length 140 \
	-visible_gpus 0 \
