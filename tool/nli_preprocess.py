#coding=utf8

import sys
import re
import json

if __name__ == '__main__':

	base_path=sys.argv[1]

	predictions = []
	for line in open(base_path+'.candidate'):
		predictions.append(line.strip())

	# process table
	inputs = []
	for line in open(base_path+'.raw_src'):
		triples = line.strip().split('\t')
		triples = [triple.replace('<SUB> ', '').replace(' <PRED> ', '|||').replace(' <OBJ> ', '|||') for triple in triples]
		new_triples = []
		for triple in triples:
			triple = triple.replace('<SUB> ', '').replace(' <PRED> ', '|||').replace(' <OBJ> ', '|||')
			sub = triple.split('|||')[0]
			predicate = triple.split('|||')[1]
			obj = triple.split('|||')[2]
			predicate = predicate.replace('-Pred-', '')	
			triple = sub + '|' + predicate + '|' + obj
			new_triples.append(triple)
		inputs.append(new_triples)

	fpout = open(base_path+'.nli', 'w')
	for (pred, in_str) in zip(predictions, inputs):
		fpout.write(json.dumps({'prediction':pred, 'input_triples':in_str})+'\n')
	fpout.close()

		
