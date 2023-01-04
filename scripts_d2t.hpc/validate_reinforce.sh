#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

MODEL_PATH=${BASE_PATH}/model.re.base/
#DATA_PATH=${BASE_PATH}/data.re.base/
DATA_PATH=${BASE_PATH}/data.re.graph_gt/

#MODEL_PATH=${BASE_PATH}/model.re.pure_merge/
#DATA_PATH=${BASE_PATH}/data.re.merge/
#DATA_PATH=${BASE_PATH}/data.re.graph_gt/

#MODEL_PATH=${BASE_PATH}/model.re.generator.rule_based/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/

LOG_PATH=${BASE_PATH}/logs.re/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py \
	-mode validate \
	-input_path ${DATA_PATH} \
	-model_path ${MODEL_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-result_path ${LOG_PATH}/validation.res \
	-log_file ${LOG_PATH}/validation.log \
	-ext_or_abs reinforce \
	-pretrain_encoder_decoder True \
	-train_predicate_graph_only False \
	-conditional_decoder True \
	-shuffle_src False \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
