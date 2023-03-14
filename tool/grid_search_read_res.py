#coding=utf8

import sys, os

dirpath = sys.argv[1]

if __name__ == '__main__':
    filelist = os.listdir(dirpath)
    filelist = sorted(filelist)
    for filename in filelist:
        filepath = './'+dirpath+'/'+filename
        #print (filepath)
        #os.system("grep BLEU " + filepath + " | awk -F ' ' '{print $3}' | sort -nr | head -1")
        hyper_bleu = {}
        for line in open(filepath):
            line = line.strip()
            if line.startswith('====== test_graph_selection_threshold'):
                hyper_parameter = line.split(' ')[2]
            if line.startswith('BLEU = '):
                bleu = line.split(', ')[0].replace('BLEU = ', '')
                hyper_bleu[hyper_parameter] = float(bleu)
        hyper, bleu = sorted(hyper_bleu.items(), key = lambda d:d[1], reverse=True)[0]
        print (filepath, f'{bleu}({hyper})')
