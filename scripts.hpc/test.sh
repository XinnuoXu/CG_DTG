#!/bin/bash

BERT_DATA_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/data/
MODEL_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/models/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_15000.pt \
	-result_path ./logs/test.res \
	-log_file ./logs/test.log \
	-use_interval true \
	-block_trigram true \
	-max_pos 512 \
	-batch_size 6000 \
	-select_topn 0.3 \
	-visible_gpus 0 \
