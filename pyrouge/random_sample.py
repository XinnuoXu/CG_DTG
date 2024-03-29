#coding=utf8

import random
import os, sys
import json

SAMPLE_NUM=15

def check_s2s_quality(print_gold=False, example_ids=None, categories=[]):
	def select_categories(system_outputs, system_inputs, system_eids, golds, categories):
		new_system_outputs = []
		new_system_inputs = []
		new_system_eids = []
		new_golds = []
		for i, eid in enumerate(system_eids):
			if eid.split('_')[-1] not in categories:
				continue
			new_system_outputs.append(system_outputs[i])
			new_system_inputs.append(system_inputs[i])
			new_system_eids.append(system_eids[i])
			new_golds.append(golds[i])
		return new_system_outputs, new_system_inputs, new_system_eids, new_golds

	#base_file = './outputs.ama/logs.summarizer/test.res.300000'
	base_file = './outputs.ama/logs.summarizer.v3.pro_or_con/test.res.100000'
	#base_file = './outputs.ama/logs.summarizer.sentence_level/test.res.100000'
	fpout = open(f'{base_file}.candidate')
	fpgold = open(f'{base_file}.gold')
	fpin = open(f'{base_file}.raw_src')
	fpeid = open(f'{base_file}.eid')
	system_outputs = [line.strip() for line in fpout]
	system_inputs = [line.strip() for line in fpin]
	system_eids = [line.strip() for line in fpeid]
	golds = [line.strip() for line in fpgold]
	if example_ids != None:
		new_system_outputs = []
		new_system_inputs = []
		new_system_eids = []
		new_golds = []
		for i in range(len(system_eids)):
			if system_eids[i] not in example_ids:
				continue
			new_system_outputs.append(system_outputs[i])
			new_system_inputs.append(system_inputs[i])
			new_system_eids.append(system_eids[i])
			new_golds.append(golds[i])
		sampled_pairs = list(zip(new_system_inputs, new_system_outputs, new_system_eids, new_golds))
	else:
		if len(categories) > 0:
			system_outputs, system_inputs, system_eids, golds = select_categories(system_outputs, system_inputs, system_eids, golds, categories)
		sampled_pairs = random.sample(list(zip(system_inputs, system_outputs, system_eids, golds)), SAMPLE_NUM)
	for pair in sampled_pairs:
		system_input = pair[0]
		system_output = pair[1]
		system_eid = pair[2]
		gold = pair[3]
		fp = open(f'./temp/{system_eid}.output', 'w')
		fp.write(f'[{system_eid}]:{system_output}')
		fp.close()
		fp = open(f'./temp/{system_eid}.input', 'w')
		fp.write(system_input.replace(' <s> ', '\n'))
		fp.close()
		fp = open(f'./temp/{system_eid}.gold', 'w')
		fp.write(gold)
		fp.close()


def check_repetitiveness():
	output_file = './outputs.ama/logs.summarizer.model_selection/test.res.120000.candidate'
	eid_file = './outputs.ama/logs.summarizer.model_selection/test.res.120000.eid'
	examples = process_system_output(output_file, eid_file)

	example_ids = examples.keys()
	sampled_examples = random.sample(example_ids, SAMPLE_NUM)

	fp = open(f'./temp/sampled.system_outputs', 'w')
	for example_id in sampled_examples:
		example = examples[example_id]
		for cate in example:
			fp.write(f'[{cate}] {example[cate]}\n')
		fp.write('\n')
	fp.close()


def check_faithfulness(example_ids=[]):
	selsum_path = '../Plan_while_Generate/AmaSum/SelSum/'
	gold_path = '../Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl'
	#system_output_file = './outputs.ama/logs.summarizer.model_selection/test.res.120000.candidate'
	#system_output_eid = './outputs.ama/logs.summarizer.model_selection/test.res.120000.eid'
	system_output_file = './outputs.ama/logs.summarizer.v3/test.res.100000.candidate'
	system_output_eid = './outputs.ama/logs.summarizer.v3/test.res.100000.eid'

	selsum_examples = process_selsum(selsum_path)
	system_examples = process_system_output(system_output_file, system_output_eid)
	gold_examples = process_gold_summary(gold_path)
	src_examples = process_src(gold_path)

	if len(example_ids) == 0:
		example_ids = src_examples.keys()
		sampled_examples = random.sample(example_ids, SAMPLE_NUM)
	else:
		sampled_examples = example_ids

	output_dir = './temp/'
	for example_id in sampled_examples:
		fpsrc = open(f'{output_dir}/{example_id}_src.txt', 'w')
		src = src_examples[example_id]
		fpsrc.write(src)
		fpsrc.close()

		fptgt = open(f'{output_dir}/{example_id}_tgt.txt', 'w')
		fptgt.write('[TARGETS]\n')
		for cate in gold_examples[example_id]:
			fptgt.write(f'[{cate}] {gold_examples[example_id][cate]}\n')
		fptgt.write('\n[SelSUM]\n')
		for cate in selsum_examples[example_id]:
			fptgt.write(f'[{cate}] {selsum_examples[example_id][cate]}\n')
		fptgt.write('\n[SYSTEM]\n')
		for cate in system_examples[example_id]:
			fptgt.write(f'[{cate}] {system_examples[example_id][cate]}\n')
		fptgt.close()


def process_src(src_filename):
	objs = [json.loads(line.strip()) for line in open(src_filename)]
	examples = {}
	for item in objs:
		eid = item['example_id']
		src = '\n'.join(item['document_segs'])
		examples[eid] = src
	return examples


def process_gold_summary(gold_filename):
	objs = [json.loads(line.strip()) for line in open(gold_filename)]
	examples = {}
	for item in objs:
		eid = item['example_id']
		gold = item['raw_tgt'].lower()
		gold_list = gold.split(' </s> ')
		examples[eid] = {}
		examples[eid]['verdict'] = gold_list[0]
		examples[eid]['pros'] = gold_list[1]
		examples[eid]['cons'] = gold_list[2]
	return examples


def process_system_output(output_file, eid_file):
	fpout = open(output_file)
	fpeid = open(eid_file)
	system_outputs = [line.strip() for line in fpout]
	system_eids = [line.strip() for line in fpeid]
	pairs = zip(system_outputs, system_eids)
	examples = {}
	for pair in pairs:
		output = ':'.join(pair[0].split(':')[1:])
		eid = pair[1]
		eid_list = eid.split('_')
		example_id = '_'.join(eid_list[:2])
		cate_id = eid_list[-1]
		if example_id not in examples:
			examples[example_id] = {}
		if cate_id not in examples[example_id]:
			examples[example_id][cate_id] = output
		else:
			examples[example_id][cate_id] += (' ' + output)
	return examples


def process_selsum(dir_path):
	res = {}
	cons = [line.strip().lower() for line in open(f'{dir_path}/test.cons')]
	pros = [line.strip().lower() for line in open(f'{dir_path}/test.pros')]
	verds = [line.strip().lower() for line in open(f'{dir_path}/test.verd')]
	triples = list(zip(verds, pros, cons))
	for i, triple in enumerate(triples):
		example_id = f'TEST_{i}'
		verd = triple[0]
		pro = triple[1]
		con = triple[2]
		res[example_id] = {}
		res[example_id]['verdict'] = verd
		res[example_id]['pros'] = pro
		res[example_id]['cons'] = con
	return res


if __name__ == '__main__':
	example_ids = set([line.strip() for line in open('./pyrouge/example_list.txt')])
	#check_s2s_quality(print_gold=True, categories=['cons'])
	#check_s2s_quality(print_gold=True, example_ids=example_ids)
	#check_repetitiveness()
	check_faithfulness(example_ids=example_ids)
