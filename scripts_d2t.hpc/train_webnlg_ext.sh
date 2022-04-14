#!/bin/bash

BERT_DATA_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/data/
MODEL_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/models.ext/
LOG_PATH=/home/hpcxu1/Planning/Tree_enc_dec/outputs.webnlg/logs.ext/

mkdir -p ${MODEL_PATH}
mkdir -p ${LOG_PATH}

python train.py  \
        -input_path ${BERT_DATA_PATH} \
        -model_path ${MODEL_PATH} \
        -tokenizer_path ${BERT_DATA_PATH}/tokenizer.pt \
        -mode train \
        -ext_or_abs ext \
        -content_planning_model tree \
        -log_file ${LOG_PATH}/train.log \
        -train_steps 60000 \
        -save_checkpoint_steps 20000 \
        -warmup_steps 1000 \
        -batch_size 3000 \
        -report_every 100 \
        -max_pos 150 \
        -max_tgt_len 150 \
        -ext_dropout 0.1 \
        -lr 0.01 \
        -accum_count 1 \
        -ext_layers 3 \
        -visible_gpus 0
