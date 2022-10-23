#coding=utf8

import os, sys
import json

FILE_DIR = '/rds/user/hpcxu1/hpc-work/outputs.ama/cluster2cls/'

if __name__ == '__main__':
	cluster_nsent = []
	cluster_ntoken = []
	num_verdict = 0
	num_pros = 0
	num_cons = 0
	num_pros_cons_overlap = 0
	num_of_selected_examples = 0
	num_examples = 0
	for filename in os.listdir(FILE_DIR):
		#if not filename.startswith('validation'):
		if not filename.startswith('train'):
			continue
		file_path = f'{FILE_DIR}/{filename}'
		data = [json.loads(line.strip()) for line in open(file_path)][0]
		for example in data:
			for cluster in example['clusters']:
				cluster_nsent.append(len(cluster))
				cluster_ntoken.append(sum([len(sent.split()) for sent in cluster]))
			num_verdict += sum(example['verdict_labels'])
			num_pros += sum(example['pros_labels'])
			num_cons += sum(example['cons_labels'])
			num_examples += len(example['cons_labels'])
			num_pros_cons_overlap += sum([example['pros_labels'][i] * example['cons_labels'][i] for i in range(len(example['pros_labels']))])
			num_of_selected_examples += sum([(example['verdict_labels'][i] | example['pros_labels'][i] | example['cons_labels'][i]) for i in range(len(example['pros_labels']))])

	print (f'Avg nsent:{sum(cluster_nsent)/len(cluster_nsent)}')
	print (f'Max nsent:{max(cluster_nsent)}')

	print (f'Avg ntoken:{sum(cluster_ntoken)/len(cluster_ntoken)}')
	print (f'Max ntoken:{max(cluster_ntoken)}')

	print (f'Number of examples: {num_examples}')
	print (f'Number of selected examples: {num_of_selected_examples}')
	print (f'Number of verdict examples: {num_verdict}')
	print (f'Number of pros examples: {num_pros}')
	print (f'Number of cons examples: {num_cons}')
	print (f'Number of pros-cons-overlap examples: {num_pros_cons_overlap}')
