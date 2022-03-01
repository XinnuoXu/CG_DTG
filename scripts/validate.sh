#!/bin/bash

BERT_DATA_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/data/
MODEL_PATH=/scratch/xxu/Plan_while_Generate/BertSumExt/xsum_bertscore/models/

python train.py \
	-mode validate \
	-input_path ${BERT_DATA_PATH} \
	-model_path ${MODEL_PATH} \
	-result_path ./logs/validation.res \
	-log_file ./logs/validation.log \
	-test_all \
	-max_pos 512 \
	-batch_size 6000 \
	-use_interval true \
	-visible_gpus 1 \
