#coding=utf8

import sys
import random

models_path = {'base': './outputs.webnlg/logs.base/test.res.3000', 
                'spectral': './outputs.webnlg/logs.partial_src/test.res'}
output_path_base = './outputs.webnlg/temp.txt'
sample_num = 20
min_ntriple = 3
max_ntriple = 4

def one_file(dataset_name):
    data_path = models_path[dataset_name]
    raw_src = data_path + '.raw_src'
    raw_gold = data_path + '.gold'
    raw_cand = data_path + '.candidate'
    raw_eid = data_path + '.eid'
    #raw_prompt = data_path + '.prompt_str'

    srcs = [line.strip().replace('<pad>', '').replace('</s>', '').replace('<s>', ' ') for line in open(raw_src)]
    golds = [line.strip() for line in open(raw_gold)]
    cands = [line.strip() for line in open(raw_cand)]
    #prompts = [line.strip() for line in open(raw_prompt)]
    eids = [line.strip().split('_')[0] for line in open(raw_eid)]

    examples = {}
    for i in range(len(srcs)):
        src = srcs[i]
        gold = golds[i]
        cand = cands[i]
        #prompt = prompts[i]
        eid = eids[i]
        #example = ' '.join(src) + '\t' + prompt + '\t' + cand + '\t' + gold
        example = [src, cand, gold]
        examples[eid] = example

    return examples


if __name__ == '__main__':

    models_results = {}
    for key in models_path:
        examples = one_file(key)
        models_results[key] = examples

    candidate_eids = []
    for eid in models_results['base']:
        if len(models_results['base'][eid][0].split('<SUB>'))-1 >= min_ntriple and \
           len(models_results['base'][eid][0].split('<SUB>'))-1 <= max_ntriple:
            candidate_eids.append(eid)
    sample_eids = random.sample(candidate_eids, sample_num)

    fpout = open(output_path_base, 'w')
    fpout.write('\t'.join(['eid', 'src', 'gold'] + list(models_results.keys())) + '\n')
    for eid in sample_eids:
        src = models_results['base'][eid][0]
        gold = models_results['base'][eid][2]
        fpout.write('\t'.join([eid, src, gold] + [models_results[key][eid][1] for key in models_results]) + '\n')
    fpout.close()
