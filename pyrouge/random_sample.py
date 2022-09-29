#coding=utf8

import random
import os, sys

SAMPLE_NUM=20

def check_s2s_quality(print_gold=False):
	fpout = open('./outputs.ama/logs.summarizer/test.res.120000.candidate')
	fpgold = open('./outputs.ama/logs.summarizer/test.res.120000.gold')
	fpin = open('./outputs.ama/logs.summarizer/test.res.120000.raw_src')
	fpeid = open('./outputs.ama/logs.summarizer/test.res.120000.eid')
	system_outputs = [line.strip() for line in fpout]
	system_inputs = [line.strip() for line in fpin]
	system_eids = [line.strip() for line in fpeid]
	golds = [line.strip() for line in fpgold]
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

if __name__ == '__main__':
	check_s2s_quality(print_gold=True)
