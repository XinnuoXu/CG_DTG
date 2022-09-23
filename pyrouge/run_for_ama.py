#coding=utf8

import sys
from utils import rouge_results_to_str, test_rouge
from nltk import word_tokenize
#import nltk.data
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

TEMP_DIR = './temp/'
CAN_PATH = './outputs.ama/logs.summarizer/test.res.120000.candidate'
GOLE_PATH = './outputs.ama/logs.summarizer/test.res.120000.gold'
EID_PATH = './outputs.ama/logs.summarizer/test.res.120000.eid'

def run_one_category(categroy, pairs):

	gold_path = TEMP_DIR + categroy + '.gold'
	cand_path = TEMP_DIR + categroy + '.cand'
	fpout_gold = open(gold_path, 'w')
	fpout_cand = open(cand_path, 'w')

	for key in pairs:
		if not key.endswith(categroy):
			continue
		fpout_gold.write(pairs[key][0]+'\n')
		fpout_cand.write(pairs[key][1]+'\n')

	fpout_gold.close()
	fpout_cand.close()

	return test_rouge(TEMP_DIR, cand_path, gold_path)

if __name__ == '__main__':
	candidates = [' '.join(word_tokenize(line.strip())) for line in open(CAN_PATH)]
	golds = [line.strip() for line in open(GOLE_PATH)]
	eids = [line.strip() for line in open(EID_PATH)]
	triples = zip(eids, golds, candidates)

	pairs = {}
	for triple in triples:
		eid = triple[0]
		gold = triple[1]
		cand = triple[2]

		eid_list = eid.split('_')
		example_id = '_'.join(eid_list[:2])
		tag = cand.split(' ')[0]
		line_id = example_id + '_' + tag
	
		#cand = cand.split(' : ')[1]
		if line_id not in pairs:
			pairs[line_id] = [gold, cand]	
		else:
			pairs[line_id][0] += (' ' + gold)
			pairs[line_id][1] += (' ' + cand)
	
	pros_results_dict = run_one_category('pros', pairs)
	'''
	cons_results_dict = run_one_category('cons', pairs)
	verdict_results_dict = run_one_category('verdict', pairs)

	print ('VERDICT results')
	print (rouge_results_to_str(verdict_results_dict))
	print ('PROS results')
	print (rouge_results_to_str(pros_results_dict))
	print ('CONS results')
	print (rouge_results_to_str(cons_results_dict))

	'''
