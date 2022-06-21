#coding=utf8

#file_dir = './outputs.webnlg/logs.single_sentences/test.res.5000'
#target_dir = './outputs.webnlg/logs.single_sentences/test.res'

#file_dir = './outputs.webnlg/logs.step_wise_parallel/test.res.6000'
#target_dir = './outputs.webnlg/logs.step_wise_parallel/test.res'

file_dir = './outputs.webnlg/logs.src_prompt_parallel/test.res.5000'
target_dir = './outputs.webnlg/logs.src_prompt_parallel/test.res'

#file_dir = './outputs.webnlg/logs.tgt_prompt_parallel/test.res.8000'
#target_dir = './outputs.webnlg/logs.tgt_prompt_parallel/test.res'

max_num_of_sentences = 10

if __name__ == '__main__':
    eids = [line.strip() for line in open(file_dir+'.eid')]
    golds = [line.strip() for line in open(file_dir+'.gold')]
    cands = [line.strip() for line in open(file_dir+'.candidate')]

    fpout_eid = open(target_dir+'.eid', 'w')
    fpout_gold = open(target_dir+'.gold', 'w')
    fpout_cand = open(target_dir+'.candidate', 'w')

    example_golds = {}
    example_cands = {}
    for i, eid in enumerate(eids):
        example_id = '_'.join(eid.split('_')[:-1])
        sentence_id = int(eid.split('_')[-1])
        gold_sent = golds[i]
        cand_sent = cands[i]
        if example_id not in example_golds:
            example_golds[example_id] = gold_sent
            example_cands[example_id] = [''] * max_num_of_sentences
        example_cands[example_id][sentence_id] = cand_sent

    for example_id in example_golds:
        fpout_eid.write(example_id + '\n')
        fpout_gold.write(example_golds[example_id] + '\n')
        fpout_cand.write(' '.join([sent for sent in example_cands[example_id] if sent != '']).strip() + '\n')

    fpout_eid.close()
    fpout_gold.close()
    fpout_cand.close()
