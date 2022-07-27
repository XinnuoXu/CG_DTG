#coding=utf8

import random

if __name__ == '__main__':
    gold_path = './outputs.webnlg/logs.partial_src/test.res.gold'
    target_path = './outputs.webnlg/logs.partial_src/test.res.candidate'
    fpout = open(target_path, 'w')
    for line in open(gold_path):
        line = line.strip().split('<ref-sep> ')[1].strip()
        flist = line.split('<q>')
        random.shuffle(flist)
        fpout.write(' '.join(flist) + '\n')
    fpout.close()
