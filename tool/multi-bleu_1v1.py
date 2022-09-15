#coding=utf8

import sys, os
import nltk.data
path_prefix = sys.argv[1] #'./outputs.webnlg/logs.base/test.res.5000'

def postprocess(string, is_ref):
    if is_ref:
        string = string.replace(' .', '.').replace(' , ', ', ')
    string = string.replace('.', ' . ').replace(', ', ' , ').replace('\'', ' \' ').replace('/', ' / ')
    string = string.replace('(', ' ( ').replace(')', ' ) ').replace('-', ' - ').replace('\"', ' \" ').replace('  ', ' ')
    return string.strip()

def process(referece, candidate):

    fpout_cand = open(path_prefix+'.bleu_cand', 'w')
    fpout_ref1 = open(path_prefix+'.bleu_ref1', 'w')
    fpout_ref2 = open(path_prefix+'.bleu_ref2', 'w')
    fpout_ref3 = open(path_prefix+'.bleu_ref3', 'w')

    reference = referece.replace('<q>', ' ')
    candidate = candidate.replace('<q>', ' ')

    reference = postprocess(reference, True)
    candidate = postprocess(candidate, False)
    ref = reference.split('<ref - sep> ')[1:]
    #ref = reference.split('<ref-sep> ')[1:]

    fpout_cand.write(candidate+'\n')
    fpout_ref1.write(ref[0].strip()+'\n')
    if len(ref) > 1:
        fpout_ref2.write(ref[1].strip()+'\n')
    else:
        fpout_ref2.write('\n')
    if len(ref) > 2:
        fpout_ref3.write(ref[2].strip()+'\n')
    else:
        fpout_ref3.write('\n')

    fpout_cand.close()
    fpout_ref1.close()
    fpout_ref2.close()
    fpout_ref3.close()

    ref1_path = path_prefix + '.bleu_ref1'
    ref2_path = path_prefix + '.bleu_ref2'
    ref3_path = path_prefix + '.bleu_ref3'
    cand_path = path_prefix + '.bleu_cand'
    os.system('./tool/multi-bleu.perl %s %s %s < %s' % (ref1_path, ref2_path, ref3_path, cand_path))


if __name__ == '__main__':
    refereces = [line.strip() for line in open(path_prefix+'.gold')]
    candidates = [line.strip() for line in open(path_prefix+'.candidate')]
    for i in range(len(refereces)):
        process(refereces[i], candidates[i])