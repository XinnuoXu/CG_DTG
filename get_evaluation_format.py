#coding=utf8

import re
import os
import sys
import json

def format_for_human_eva(srcs, refs, candidates):
    pass

def format_for_auto_eva(srcs, refs, candidates):
    source_json = {"values":[], "language":"en"}
    reference_json = {"values":[], "language":"en"}
    prediction_json = {"values":[], "language":"en"}

    for idx, src in enumerate(srcs):
        ref = refs[idx]
        cand = candidates[idx]
        reference_json["values"].append({"target":[ref]})
        prediction_json["values"].append(cand)
        source_json["values"].append(src)

    ref_fpout = open('./temp/references.json', 'w')
    pred_fpout = open('./temp/predictions.json', 'w')
    source_fpout = open('./temp/sources.json', 'w')

    ref_fpout.write(json.dumps(reference_json))
    pred_fpout.write(json.dumps(prediction_json))
    source_fpout.write(json.dumps(source_json))

    ref_fpout.close()
    pred_fpout.close()
    source_fpout.close()

if __name__ == '__main__':

    base_path = sys.argv[1] # something like './outputs/logs.xsum.bartbase/test.res.100000'
    eva_mode = sys.argv[2] # auto or human

    src_path = base_path + ".raw_src"
    gold_summ_path = base_path + ".gold"
    candid_path = base_path + ".candidate"

    #src_path = os.path.join(input_dir, 'test.res.src')
    #gold_summ_path = os.path.join(input_dir, 'test.res.gold')
    #candid_path = os.path.join(input_dir, 'test.res.candidate')

    srcs = [line.strip() for line in open(src_path)]
    gold_summs = [line.strip() for line in open(gold_summ_path)]
    candidates = [line.strip() for line in open(candid_path)]
    
    if eva_mode == 'auto':
        format_for_auto_eva(srcs, gold_summs, candidates)
    elif eva_mode == 'human':
        format_for_human_eva(srcs, gold_summs, candidates)
    else:
        print ('Can not find eva_mode (\'auto\' or \'human\'). ')
