#coding=utf8

import json

if __name__ == '__main__':
    num_of_examples = 0
    bad_examples = 0
    bad_examples_more_sentences = 0
    num_of_predicates = {}
    num_of_sentences = {}
    matrix = [[0]*8 for i in range(7)]
    for line in open('../Plan_while_Generate/D2T_data/webnlg_data/train.jsonl'):
        obj = json.loads(line.strip())
        predicates = obj['predicates']
        alignments = obj['oracles_selection']
        if sum([len(item) for item in alignments]) < len(predicates):
            bad_examples += 1
            if len(alignments) > 1:
                bad_examples_more_sentences += 1
        num_of_examples += 1
        if len(predicates) not in num_of_predicates:
            num_of_predicates[len(predicates)] = 1
        else:
            num_of_predicates[len(predicates)] += 1
        if len(alignments) not in num_of_sentences:
            num_of_sentences[len(alignments)] = 1
        else:
            num_of_sentences[len(alignments)] += 1
        matrix[len(alignments)][len(predicates)] += 1
    print ('number of predicates:')
    print (num_of_predicates)
    print ('number of sentences:')
    print (num_of_sentences)
    print (matrix)

    print (bad_examples, bad_examples_more_sentences, num_of_examples, bad_examples/num_of_examples)
