#coding=utf8

import json
import sys
import os

if __name__ == '__main__':
    heights = []
    #for line in open('outputs.cnn_dm/logs.freeze_tmt/test.res.320000.trees_gumble'):
    #for line in open('outputs.cnn_dm/logs.freeze_tmt/test.res.320000.trees'):
    for line in open('outputs.cnn_dm/logs.freeze_tmt/test.res.320000.trees_free'):
        tree = json.loads(line.strip())
        heights.append(tree['Height'])
    print (sum(heights)/len(heights))
