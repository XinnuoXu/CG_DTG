#coding=utf8

if __name__ == '__main__':
    ids = [line.strip() for line in open('selected_ids.txt')]
    cands = [line.strip() for line in open('outputs.webnlg/logs.partial_src/test.res.candidate')]
    eids = [line.strip() for line in open('outputs.webnlg/logs.partial_src/test.res.eid')]
    id_to_cand = {}
    for i, eid in enumerate(eids):
        id_to_cand[eid] = cands[i]
    for id in ids:
        print (id_to_cand[id])
