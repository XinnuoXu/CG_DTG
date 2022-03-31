#coding=utf8

import json
import sys
import os

if __name__ == '__main__':
    heights = []
    #for line in open('outputs.cnn_dm/logs.freeze_tmt/test.res.320000.trees_gumble'):
    #for line in open('outputs.cnn_dm/logs.freeze_tmt/test.res.320000.trees_free'):
    fpout = open('./outputs.cnn_dm/logs.freeze_tmt/tree.txt', 'w')
    for line in open('outputs.cnn_dm/logs.freeze_tmt/test.res.320000.trees'):
        tree = json.loads(line.strip())
        heights.append(tree['Height'])
        tree_str = tree['Tree']
        src_sents = tree['Src']
        fpout.write(tree_str + '\n')
        fpout.write('\n'.join(src_sents) + '\n\n')
    print (sum(heights)/len(heights))
    fpout.close()
    
