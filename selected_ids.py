#coding=utf8

candidate_path = 'outputs.webnlg/logs.partial_src/test.res.candidate'
eid_path = 'outputs.webnlg/logs.partial_src/test.res.eid'

#candidate_path = './outputs.webnlg/logs.prefix_tgt/test.res.4000.candidate'
#eid_path = './outputs.webnlg/logs.prefix_tgt/test.res.4000.eid'

if __name__ == '__main__':
    ids = [line.strip() for line in open('selected_ids.txt')]
    cands = [line.strip() for line in open(candidate_path)]
    eids = [line.strip().split('_')[0] for line in open(eid_path)]
    id_to_cand = {}
    for i, eid in enumerate(eids):
        id_to_cand[eid] = cands[i]
    for id in ids:
        print (id_to_cand[id])
