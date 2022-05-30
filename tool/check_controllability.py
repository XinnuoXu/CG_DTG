#coding=utf8

import random

data_path = './outputs.webnlg/logs.src_prompt/test.res.4000'
output_path = './outputs.webnlg/temp.txt'

if __name__ == '__main__':
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
