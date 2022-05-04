#coding=utf8

import sys, os

path_prefix = sys.argv[1] #'./outputs.webnlg/logs.base/test.res.5000'
target_format = sys.argv[2] # ['one-to-one', 'first-to-many', 'first-to-first']

def first_to_many(example_ids, refereces, candidates):

    res = {}
    for i, example_id in enumerate(example_ids):
        reference = refereces[i].replace('<q>', ' ')
        candidate = candidates[i].replace('<q>', ' ')
        eid, rid = example_id.split('_')
        if eid not in res:
            res[eid] = {'ref':[], 'cand':''}
        if rid == 'Id1':
            res[eid]['cand'] = candidate
        if rid in ['Id1', 'Id2', 'Id3']:
            res[eid]['ref'].append(reference)

    fpout_cand = open(path_prefix+'.bleu_cand', 'w')
    fpout_ref1 = open(path_prefix+'.bleu_ref1', 'w')
    fpout_ref2 = open(path_prefix+'.bleu_ref2', 'w')
    fpout_ref3 = open(path_prefix+'.bleu_ref3', 'w')

    for eid in res:
        ref = res[eid]['ref']
        cand = res[eid]['cand']
        fpout_cand.write(cand+'\n')
        fpout_ref1.write(ref[0]+'\n')
        if len(ref) > 1:
            fpout_ref2.write(ref[1]+'\n')
        else:
            fpout_ref2.write('\n')
        if len(ref) > 2:
            fpout_ref3.write(ref[2]+'\n')
        else:
            fpout_ref3.write('\n')

    fpout_cand.close()
    fpout_ref1.close()
    fpout_ref2.close()
    fpout_ref3.close()


if __name__ == '__main__':
    example_ids = [line.strip() for line in open(path_prefix+'.eid')]
    refereces = [line.strip() for line in open(path_prefix+'.gold')]
    candidates = [line.strip() for line in open(path_prefix+'.candidate')]
    if target_format == 'first-to-many':
        first_to_many(example_ids, refereces, candidates)
    
