#coding=utf8

import json
import sys
#sys.path.append('../../')
from models import alignment

align_obj = alignment.InputOutputAlignment(run_rouge=False, run_bleu=True, run_bertscore=False, run_bleurt=False, coreference=True)

def align(filename, output_filename):

    fpout = open(output_filename, 'w')

    for line in open(filename):
        json_obj = json.loads(line.strip())
        srcs = json_obj['document_segs']

        tgts = json_obj['gold_segs']
        if len(tgts) < 2:
            alignments = [[i for i in range(0, len(srcs))]]
        else:
            alignments = align_obj.input_to_output(srcs, tgts)

        json_obj['oracles_selection'] = alignments
        fpout.write(json.dumps(json_obj)+'\n')

    fpout.close()

if __name__ == '__main__':

    

