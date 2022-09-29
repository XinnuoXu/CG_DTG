#coding=utf8

import os, sys
import json
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


def process_selsum(candidate_dir):
	cons = [line.strip().lower() for line in open(candidate_dir+'/test.cons')]
	pros = [line.strip().lower() for line in open(candidate_dir+'/test.pros')]
	verds = [line.strip().lower() for line in open(candidate_dir+'/test.verd')]
	id_to_sentences = {}
	for idx, con in enumerate(cons):
		id_to_sentences[f"TEST_{idx}_cons"] = con
		id_to_sentences[f"TEST_{idx}_pros"] = pros[idx]
		id_to_sentences[f"TEST_{idx}_verdict"] = verds[idx]
	return id_to_sentences


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


def process_raw(gold_path):
	id_to_sentences = {}
	json_objs = [json.loads(line.strip()) for line in open(gold_path)]
	for json_obj in json_objs:
		raw_tgt = json_obj['raw_tgt']
		example_id = json_obj['example_id']
		summaries = raw_tgt.split(' </s> ')
		verdict = summaries[0]
		pros = summaries[1]
		cons = summaries[2]
		id_to_sentences[f"{example_id}_verdict"] = verdict.lower()
		id_to_sentences[f"{example_id}_cons"] = cons.lower()
		id_to_sentences[f"{example_id}_pros"] = pros.lower()
	return id_to_sentences


def process_semantic_clean(gold_path):
	id_to_sentences = {}
	json_objs = []
	for filename in os.listdir(gold_path):
	    file_path = os.path.join(gold_path, filename)
	    json_objs.extend([json.loads(line.strip()) for line in open(file_path)])
	for json_obj in json_objs:
		tgt_sentences = json_obj['cleaned_tgt']
		example_id = json_obj['example_id']
		verdicts = []; pros = []; cons = []
		for sentence in tgt_sentences:
			chunks = sentence.split(':')
			summary_sentence = ':'.join(chunks[1:]).strip()
			summary_type = sentence.split(' ')[0]
			if summary_type == 'verdict':
				verdicts.append(summary_sentence)
			elif summary_type == 'pros':
				pros.append(summary_sentence)
			elif summary_type == 'cons':
				cons.append(summary_sentence)
		if len(verdicts) > 0:
			id_to_sentences[f"{example_id}_verdict"] = ' '.join(verdicts).lower()
		if len(cons) > 0:
			id_to_sentences[f"{example_id}_cons"] = ' '.join(cons).lower()
		if len(pros) > 0:
			id_to_sentences[f"{example_id}_pros"] = ' '.join(pros).lower()
	return id_to_sentences


def process_cluster_clean(gold_path):
	json_objs = []
	for filename in os.listdir(gold_path):
		if not filename.startswith('test.'):
			continue
		file_path = os.path.join(gold_path, filename)
		read_objs = [json.loads(line.strip()) for line in open(file_path)]
		for batch in read_objs:
			json_objs.extend(batch)

	id_to_sentences = {}
	for json_obj in json_objs:
		eid = json_obj['example_id']
		eid_list = eid.split('_')
		example_id = '_'.join(eid_list[:2])
		tag = eid_list[-1]
		example_id = example_id + '_' + tag

		tgt_sentences = ' '.join([' '.join(item) for item in json_obj['tgt']])
		if example_id not in id_to_sentences:
			id_to_sentences[example_id] = tgt_sentences
		else:
			id_to_sentences[example_id] += (' ' + tgt_sentences)
	return id_to_sentences


def process_cluster_clean_v1(gold_path, eid_path):
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
		if idx not in id_to_golds:
			continue
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
	elif args.cand_type == 'selsum':
		id_to_cands = process_selsum(args.cand_path)
	
	# Process golds
	if args.gold_type == 'cluster_clean':
		id_to_golds = process_cluster_clean(args.gold_path)
	elif args.gold_type == 'raw':
		id_to_golds = process_raw(args.gold_path)
	elif args.gold_type == 'clean':
		id_to_golds = process_semantic_clean(args.gold_path)

	# Run pyrouge
	run_pyrouge(id_to_cands, id_to_golds)
