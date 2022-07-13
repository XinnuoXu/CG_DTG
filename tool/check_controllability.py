#coding=utf8

import sys
import random

data_path = './outputs.webnlg/logs.tgt_prompt_parallel/test.res'
#data_path = './outputs.webnlg/logs.tgt_prompt/test.res.4000'
output_path = './outputs.webnlg/temp_tgt_para_control.txt'

def version_1():
    raw_src = data_path + '.raw_src'
    raw_gold = data_path + '.gold'
    raw_cand = data_path + '.candidate'

    srcs = [line.strip().replace('<pad>', '').replace('</s>', '').split('<s>') for line in open(raw_src)]
    golds = [line.strip() for line in open(raw_gold)]
    cands = [line.strip() for line in open(raw_cand)]

    examples = []
    for i in range(len(srcs)):
        src = srcs[i]
        gold = golds[i]
        cand = cands[i]
        if len(src) < 7:
            continue
        example = ' '.join(src[:-1]) + '\t' + src[-1] + '\t' + cand + '\t' + gold
        examples.append(example)

    random.shuffle(examples)
    sampled_examples = examples[:15]
    fpout = open(output_path, 'w')
    for example in sampled_examples:
        fpout.write(example+'\n')
    fpout.close()

def version_2():
    raw_src = data_path + '.raw_src'
    raw_alg = data_path + '.alignments'
    raw_gold = data_path + '.gold'
    raw_cand = data_path + '.candidate'

    srcs = [line.strip().replace('<pad>', '').replace('</s>', '').split('<s>') for line in open(raw_src)]
    golds = [line.strip() for line in open(raw_gold)]
    cands = [line.strip() for line in open(raw_cand)]
    algs = [line.strip() for line in open(raw_alg)]

    examples = []
    for i in range(len(srcs)):
        src = srcs[i]
        gold = golds[i]
        cand = cands[i]
        alg = algs[i]
        if len(src) < 7:
            continue
        example = ' '.join(src) + '\t' + alg + '\t' + cand + '\t' + gold
        examples.append(example)

    random.shuffle(examples)
    sampled_examples = examples[:15]
    fpout = open(output_path, 'w')
    for example in sampled_examples:
        fpout.write(example+'\n')
    fpout.close()

def version_3():
    raw_src = data_path + '.raw_src'
    prompt_str = data_path + '.prompt_str'
    raw_gold = data_path + '.gold'
    raw_cand = data_path + '.candidate'

    srcs = [line.strip().replace('<pad>', '').replace('</s>', '').split('<s>') for line in open(raw_src)]
    golds = [line.strip() for line in open(raw_gold)]
    cands = [line.strip() for line in open(raw_cand)]
    prompts = [line.strip() for line in open(prompt_str)]

    examples = []
    for i in range(len(srcs)):
        src = srcs[i]
        gold = golds[i]
        cand = cands[i]
        prompt = prompts[i]
        if len(src) < 4:
            continue
        example = ' '.join(src) + '\t' + prompt + '\t' + cand + '\t' + gold
        examples.append(example)

    random.shuffle(examples)
    sampled_examples = examples[:15]
    fpout = open(output_path, 'w')
    for example in sampled_examples:
        fpout.write(example+'\n')
    fpout.close()

if __name__ == '__main__':
    if sys.argv[1] == 'v1':
        version_1()
    if sys.argv[1] == 'v2':
        version_2()
    else:
        version_3()
