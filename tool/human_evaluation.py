#coding=utf8

import sys
import random

models_path = {'tgt_prompt': './outputs.webnlg/logs.tgt_prompt/test.res.4000',
               'tgt_parallel': './outputs.webnlg/logs.tgt_prompt_parallel/test.res'}
#'base': './outputs.webnlg.bak/logs.base/test.res.3000'}
output_path_base = './outputs.webnlg/temp'
sample_num = 20

def one_file(dataset_name):
    data_path = models_path[dataset_name]
    raw_src = data_path + '.raw_src'
    raw_gold = data_path + '.gold'
    raw_cand = data_path + '.candidate'
    raw_eid = data_path + '.eid'
    raw_prompt = data_path + '.prompt_str'

    srcs = [line.strip().replace('<pad>', '').replace('</s>', '').split('<s>') for line in open(raw_src)]
    golds = [line.strip() for line in open(raw_gold)]
    cands = [line.strip() for line in open(raw_cand)]
    prompts = [line.strip() for line in open(raw_prompt)]
    eids = [line.strip().split('_')[0] for line in open(raw_eid)]

    examples = {}
    for i in range(len(srcs)):
        src = srcs[i]
        gold = golds[i]
        cand = cands[i]
        prompt = prompts[i]
        eid = eids[i]
        if len(src) < 4:
            continue
        example = ' '.join(src) + '\t' + prompt + '\t' + cand + '\t' + gold
        examples[eid] = example

    return examples


if __name__ == '__main__':

    sample_eids = None

    for key in models_path:
        examples = one_file(key)
        if sample_eids is None:
            sample_eids = random.sample(examples.keys(), sample_num)

        output_path = output_path_base + '.' + key
        fpout = open(output_path, 'w')
        for eid in sample_eids:
            fpout.write(examples[eid]+'\n')
        fpout.close()
