#!/bin/bash

percent=$1
test_from=$2

BASE_PATH=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.bbase/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.bbase/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.bbase/
# ntriple=0.005; test_from=500
# ntriple=0.01; test_from=1000
# ntriple=0.05; test_from=5000
# ntriple=0.1; test_from=5000

#[CHOOSE]
MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.blarge/
DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.blarge/
LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.blarge/
# ntriple=0.005; test_from=500
# ntriple=0.01; test_from=1500
# ntriple=0.05; test_from=2500
# ntriple=0.1; test_from=4000

#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.t5small/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.t5small/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.t5small/
# ntriple=0.005; test_from=1500
# ntriple=0.01; test_from=3000
# ntriple=0.05; test_from=5000
# ntriple=0.1; test_from=5000

#MODEL_PATH=${BASE_PATH}/model.re.encdec_partial.t5large/
#DATA_PATH=${BASE_PATH}/data.re.merge.tokenized_preds.t5large/
#LOG_PATH=${BASE_PATH}/logs.re.encdec_partial.t5large/
# ntriple=0.005; test_from=500
# ntriple=0.01; test_from=1000
# ntriple=0.05; test_from=4000
# ntriple=0.1; test_from=5000

mkdir -p ${LOG_PATH}

python train.py \
	-mode test \
	-input_path ${DATA_PATH} \
        -tokenizer_path ${DATA_PATH}/tokenizer.pt \
	-test_from ${MODEL_PATH}/model_step_${test_from}.pt \
	-test_unseen False \
	-result_path ${LOG_PATH}/test.res \
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
