#coding=utf8

import os, sys
import json

FILE_DIR = './outputs.ama/cluster2cls/'

if __name__ == '__main__':
	cluster_nsent = []
	cluster_ntoken = []
	for filename in os.listdir(FILE_DIR):
		if not filename.startswith('validation'):
			continue
		file_path = f'{FILE_DIR}/{filename}'
		data = [json.loads(line.strip()) for line in open(file_path)][0]
		for example in data:
			for cluster in example['clusters']:
				cluster_nsent.append(len(cluster))
				cluster_ntoken.append(sum([len(sent.split()) for sent in cluster]))

	print (f'Avg nsent:{sum(cluster_nsent)/len(cluster_nsent)}')
	print (f'Max nsent:{max(cluster_nsent)}')

	print (f'Avg ntoken:{sum(cluster_ntoken)/len(cluster_ntoken)}')
	print (f'Max ntoken:{max(cluster_ntoken)}')
