#coding=utf8

import sys
import re

def camel_case_split(dentifier):
	dentifier = dentifier.replace('-Pred-', '')
	matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', dentifier)
	d = [m.group(0) for m in matches]
	new_d = []
	for token in d:
		token = token.replace('(', '')
		token_split = token.split('_')
		for t in token_split:
			new_d.append(t.lower())
			#new_d.append(t)
	return ' '.join(new_d)

if __name__ == '__main__':

	base_path=sys.argv[1]

	# process reference
	fpout = open(base_path+'.parent_ref', 'w')
	for line in open(base_path+'.gold'):
		refs = [item.strip() for item in line.strip().split('<ref-sep>')][1:]
		refs = '\t'.join(refs)
		fpout.write(refs+'\n')
	fpout.close()

	# process prediction
	fpout = open(base_path+'.parent_pred', 'w')
	for line in open(base_path+'.candidate'):
		fpout.write(line.strip()+'\n')
	fpout.close()

	# process table
	fpout = open(base_path+'.parent_table', 'w')
	for line in open(base_path+'.raw_src'):
		triples = line.strip().split('\t')
		triples = [triple.replace('<SUB> ', '').replace(' <PRED> ', '|||').replace(' <OBJ> ', '|||') for triple in triples]
		new_triples = []
		for triple in triples:
			triple = triple.replace('<SUB> ', '').replace(' <PRED> ', '|||').replace(' <OBJ> ', '|||')
			sub = triple.split('|||')[0]
			predicate = triple.split('|||')[1]
			obj = triple.split('|||')[2]
			predicate = camel_case_split(predicate)		
			triple = sub + '|||' + predicate + '|||' + obj
			new_triples.append(triple)
		triples = '\t'.join(new_triples)
		fpout.write(triples+'\n')

	fpout.close()

		
