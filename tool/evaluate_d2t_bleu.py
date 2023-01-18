#coding=utf8

import re
import sys, os
import nltk.data

path_prefix = sys.argv[1] #'./outputs.webnlg/logs.base/test.res.5000'

def postprocess(string, is_ref=False):
    '''
    if is_ref:
        string = string.replace(' .', '.').replace(' , ', ', ')
    string = string.replace('.', ' . ').replace(', ', ' , ').replace('\'', ' \' ').replace('/', ' / ')
    string = string.replace('(', ' ( ').replace(')', ' ) ').replace('-', ' - ').replace('\"', ' \" ').replace('  ', ' ')
    '''
    string = ' '.join(re.split('(\W)', string))
    string = ' '.join(string.split())
    return string.strip()

def process(refereces, candidates):

    fpout_cand = open(path_prefix+'.bleu_cand', 'w')
    fpout_ref1 = open(path_prefix+'.bleu_ref1', 'w')
    fpout_ref2 = open(path_prefix+'.bleu_ref2', 'w')
    fpout_ref3 = open(path_prefix+'.bleu_ref3', 'w')

    for i in range(len(refereces)):
        references = refereces[i].replace('<q>', ' ')
        candidate = candidates[i].replace('<q>', ' ')

        candidate = postprocess(candidate)
        ref = [postprocess(item.strip()) for item in references.split('<ref-sep>')][1:]

        '''
        references = postprocess(references, True)
        if not references.startswith('<ref'):
            ref = [references.strip()]
        else:
            ref = [r.strip() for r in references.split('<ref - sep>')[1:]]
        '''

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


if __name__ == '__main__':
    refereces = [line.strip() for line in open(path_prefix+'.gold')]
    candidates = [line.strip() for line in open(path_prefix+'.candidate')]
    process(refereces, candidates)
    
