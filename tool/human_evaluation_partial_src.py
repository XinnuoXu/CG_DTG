#coding=utf8
import random

file_dir = './outputs.webnlg/logs.partial_src/test.res.4000'
target_dir = './outputs.webnlg/logs.partial_src/sample.res'

sample_number = 20
max_num_of_sentences = 10

if __name__ == '__main__':
    eids = [line.strip() for line in open(file_dir+'.eid')]
    golds = [line.strip() for line in open(file_dir+'.gold')]
    cands = [line.strip() for line in open(file_dir+'.candidate')]
    srcs = [line.strip().replace('<pad>', '').split('<s>')[1:-1] for line in open(file_dir+'.raw_src')]

    fpout = open(target_dir, 'w')

    example_golds = {}
    example_cands = {}
    example_srcs = {}
    example_ntriple = {}
    for i, eid in enumerate(eids):
        example_id = '_'.join(eid.split('_')[:-1])
        sentence_id = int(eid.split('_')[-1])
        gold_sent = golds[i]
        cand_sent = cands[i]
        src_sent = srcs[i]
        if example_id not in example_golds:
            example_golds[example_id] = [''] * max_num_of_sentences
            example_cands[example_id] = [''] * max_num_of_sentences
            example_srcs[example_id] = [''] * max_num_of_sentences
            example_ntriple[example_id] = 0
        example_golds[example_id][sentence_id] = gold_sent
        example_cands[example_id][sentence_id] = cand_sent
        example_srcs[example_id][sentence_id] = '  '.join(src_sent)
        example_ntriple[example_id] += len(src_sent)

    long_examples = [example_id for example_id in example_ntriple if example_ntriple[example_id] > 3]

    sample_ids = random.sample(long_examples, sample_number) 
    for example_id in example_golds:
        if example_id not in sample_ids:
            continue
        fpout.write(example_id + '\n\n')
        for i, gold in enumerate(example_golds[example_id]):
            cand = example_cands[example_id][i]
            src = example_srcs[example_id][i]
            if gold == '':
                break
            fpout.write(src + '\n')
            fpout.write(cand + '\n\n')
        fpout.write(example_golds[example_id][0] + '\n\n\n')
            
    fpout.close()
