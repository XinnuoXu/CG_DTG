#coding=utf8

import sys
import argparse
from utils import rouge_results_to_str, test_rouge
from nltk import word_tokenize

TEMP_DIR = './temp/'

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def process_system_output(candidate_path, eid_path):
	candidates = [' '.join(word_tokenize(line.strip())) for line in open(candidate_path)]
	eids = [line.strip() for line in open(eid_path)]
	zipped_pairs = zip(eids, candidates)

	id_to_sentences = {}
	for pair in zipped_pairs:
		eid = pair[0]
		cand = pair[1]

		eid_list = eid.split('_')
		example_id = '_'.join(eid_list[:2])
		tag = eid_list[-1]
		line_id = example_id + '_' + tag
	
		cand = ':'.join(cand.split(':')[1:]).strip()
		if line_id not in id_to_sentences:
			id_to_sentences[line_id] = cand
		else:
			id_to_sentences[line_id] += (' ' + cand)
	
	return id_to_sentences


def process_cluster_clean(gold_path, eid_path):
	golds = [line.strip() for line in open(gold_path)]
	eids = [line.strip() for line in open(eid_path)]
	zipped_pairs = zip(eids, golds)

	id_to_sentences = {}
	for pair in zipped_pairs:
		eid = pair[0]
		gold = pair[1]

		eid_list = eid.split('_')
		example_id = '_'.join(eid_list[:2])
		tag = eid_list[-1]
		line_id = example_id + '_' + tag
	
		if line_id not in id_to_sentences:
			id_to_sentences[line_id] = gold
		else:
			id_to_sentences[line_id] += (' ' + gold)
	
	return id_to_sentences


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


def run_pyrouge(id_to_cands, id_to_golds):

	pairs = {}
	for idx in id_to_cands:
		cand = id_to_cands[idx]
		gold = id_to_golds[idx]
		pairs[idx] = [gold, cand]

	pros_results_dict = run_one_category('pros', pairs)
	cons_results_dict = run_one_category('cons', pairs)
	verdict_results_dict = run_one_category('verdict', pairs)

	print ('VERDICT results')
	print (rouge_results_to_str(verdict_results_dict))
	print ('PROS results')
	print (rouge_results_to_str(pros_results_dict))
	print ('CONS results')
	print (rouge_results_to_str(cons_results_dict))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-gold_type", default='cluster_clean', type=str, choices=['raw', 'clean', 'cluster_clean'])
	parser.add_argument("-gold_path", default='', type=str)
	parser.add_argument("-cand_type", default='systems', type=str, choices=['systems', 'selsum'])
	parser.add_argument("-cand_path", default='', type=str)
	parser.add_argument("-eid_path", default='', type=str)
	args = parser.parse_args()

	# Process system outputs
	if args.cand_type == 'systems':
		id_to_cands = process_system_output(args.cand_path, args.eid_path)
	
	# Process golds
	if args.gold_type == 'cluster_clean':
		id_to_golds = process_cluster_clean(args.gold_path, args.eid_path)

	# Run pyrouge
	run_pyrouge(id_to_cands, id_to_golds)
