#coding=utf8

#file_dir = './outputs.webnlg/logs.single_sentences/test.res.5000'
#target_dir = './outputs.webnlg/logs.single_sentences/test.res'

#file_dir = './outputs.webnlg/logs.step_wise_parallel/test.res.6000'
#target_dir = './outputs.webnlg/logs.step_wise_parallel/test.res'

#file_dir = './outputs.webnlg/logs.tgt_prompt_parallel/test.res.6000'
#target_dir = './outputs.webnlg/logs.tgt_prompt_parallel/test.res'

#file_dir = './outputs.webnlg/logs.single_sentences_src_prompts/test.res.5000'
#target_dir = './outputs.webnlg/logs.single_sentences_src_prompts/test.res'

file_dir = './outputs.webnlg/logs.partial_src/test.res.4000'
target_dir = './outputs.webnlg/logs.partial_src/test.res'

#file_dir = './outputs.webnlg/logs.partial_prompt/test.res.5000'
#target_dir = './outputs.webnlg/logs.partial_prompt/test.res'

max_num_of_sentences = 10

if __name__ == '__main__':
    eids = [line.strip() for line in open(file_dir+'.eid')]
    golds = [line.strip() for line in open(file_dir+'.gold')]
    cands = [line.strip() for line in open(file_dir+'.candidate')]
    srcs = [line.strip().replace('<pad>', '').split('<s>')[1:-1] for line in open(file_dir+'.raw_src')]

    fpout_eid = open(target_dir+'.eid', 'w')
    fpout_gold = open(target_dir+'.gold', 'w')
    fpout_cand = open(target_dir+'.candidate', 'w')
    fpout_src = open(target_dir+'.raw_src', 'w')

    example_golds = {}
    example_cands = {}
    example_srcs = {}
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
        example_golds[example_id][sentence_id] = gold_sent
        example_cands[example_id][sentence_id] = cand_sent
        example_srcs[example_id][sentence_id] = ' '.join(src_sent)

    for example_id in example_golds:
        fpout_eid.write(example_id + '\n')
        #fpout_gold.write('<q>'.join([sent for sent in example_golds[example_id] if sent != '']).strip() + '\n')
        fpout_gold.write(example_golds[example_id][0].strip() + '\n')
        fpout_cand.write(' '.join([sent for sent in example_cands[example_id] if sent != '']).strip() + '\n')
        fpout_src.write(' '.join(example_srcs[example_id]).strip() + '\n')

    fpout_eid.close()
    fpout_gold.close()
    fpout_cand.close()
    fpout_src.close()
