#!/bin/bash

percent=$1

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.bbase/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.bbase/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.bbase/

MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.blarge/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.blarge/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.blarge/

#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.t5small/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.t5small/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.t5small/

#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.t5large/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.t5large/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.t5large/

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
	-nn_graph True \
	-shuffle_src False \
        -max_pos 250 \
	-batch_size 200 \
	-max_tgt_len 250 \
	-visible_gpus 0 \
