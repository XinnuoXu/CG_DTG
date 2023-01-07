#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

#MODEL_PATH=${BASE_PATH}/model.re.from_scratch/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.from_scratch/
# 5000/40000

#MODEL_PATH=${BASE_PATH}/model.re.gold_random/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.gold_random/
# 20000

#MODEL_PATH=${BASE_PATH}/model.re.evenly_mix/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.evenly_mix/
# 20000

#MODEL_PATH=${BASE_PATH}/model.re.joint/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#LOG_PATH=${BASE_PATH}/logs.re.joint/
# 35000

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
	-pretrain_encoder_decoder False \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-shuffle_src False \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
