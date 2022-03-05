#!/bin/bash

BERT_DATA_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/data/
MODEL_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/models.greedy/
#MODEL_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/models.rouge.top1/
#MODEL_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/models.bertscore.top3/

python train.py \
	-mode test \
	-input_path ${BERT_DATA_PATH} \
	-test_from ${MODEL_PATH}/model_step_15000.pt \
	-result_path ./logs.tmp/test.res \
	-log_file ./logs.tmp/test.log \
	-test_data_source test \
	-use_interval true \
	-block_trigram true \
	-max_pos 512 \
	-batch_size 6000 \
	-select_topn 3 \
	-visible_gpus 1 \
	
	#-test_data_source validation \
