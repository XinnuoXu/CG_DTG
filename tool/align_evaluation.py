#coding=utf8

import json
import sys
sys.path.append('./')
from models import alignment

align_obj = alignment.InputOutputAlignment(run_rouge=False, run_bleu=True, run_bertscore=False, run_bleurt=False, coreference=True)

def align(src_path, tgt_path):

    srcs = [line.strip().split('\t') for line in open(src_path)]
    tgts = [[line.strip()] for line in open(tgt_path)]

    scores = []; score_dict = {}
    for i, src in enumerate(srcs):
        tgt = tgts[i]
        _, score = align_obj.input_to_output(src, tgt)
        scores.append(score)
        if len(src) not in score_dict:
            score_dict[len(src)] = []
        score_dict[len(src)].append(score)

    align_score = sum(scores)/len(scores)
    for item in sorted(score_dict.items(), key = lambda d:d[0]):
        score = sum(item[1])/len(item[1])
        print (f'Alignment scores for #{item[0]} triple: {score}')

    print (f'Alignment Scores: {align_score}')

if __name__ == '__main__':

    base_path = sys.argv[1] 
    src_path = sys.argv[1] + '.raw_src'
    tgt_path = sys.argv[1] + '.candidate'
    align(src_path, tgt_path)

