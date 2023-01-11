#!/bin/bash

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/

#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial/
#DATA_PATH=${BASE_PATH}/data.re.merge.rule_based/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial/

#MODEL_PATH=${BASE_PATH}/model.re.nn/ 
#15000/20000
#MODEL_PATH=${BASE_PATH}/model.re.nn.spectral_with_sample/
#15000/20000
MODEL_PATH=${BASE_PATH}/model.re.nn.spectral_with_sample.big_lr//
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds/
LOG_PATH=${BASE_PATH}/logs.re.nn/

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
	-nn_graph True \
	-pretrain_nn_cls True \
	-pretrain_encoder_decoder False \
	-train_predicate_graph_only True \
	-conditional_decoder True \
	-shuffle_src False \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
