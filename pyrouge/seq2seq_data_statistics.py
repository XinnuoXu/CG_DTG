#coding=utf8

import os, sys
import json
import random

FILE_DIR = '/rds/user/hpcxu1/hpc-work/outputs.ama100/seq2seq/'

if __name__ == '__main__':
	examples = []
	for filename in os.listdir(FILE_DIR):
		if not filename.startswith('train'):
			continue
		file_path = f'{FILE_DIR}/{filename}'
		data = [json.loads(line.strip()) for line in open(file_path)][0]
		for example in data:
			srcs = '\n'.join(example['src'])
			tgt = ' '.join([' '.join(sent) for sent in example['tgt']])
			tgt_prefix = ' '.join(example['tgt_prefix'])
			if tgt_prefix.startswith('verdict'):
				continue
			examples.append([srcs, tgt_prefix, tgt])
	sampled_examples = random.sample(examples, 20)
	for example in sampled_examples:
		print (example[0], '\n')
		print (example[1])
		print (example[2])
		print ('\n\n\n')
