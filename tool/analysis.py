#coding=utf8

import json
import numpy as np
from sklearn.metrics import label_ranking_loss

class Analysis():

    def __init__(self):
        from models.neural import CalculateSelfAttention
        self.self_attn_layer = CalculateSelfAttention()

    def edge_ranking_data_processing(self, sents_vec, alignments, mask_cls):
        nsent = mask_cls.sum(1)
        self_attns = self.self_attn_layer(sents_vec, sents_vec, mask_cls)
        pred_scores = []; align_labels = []

        for eid, alignment in enumerate(alignments):

            pred_score = []; align_label = []

            cluster_labels = {}; cluster_lable = 0
            for sent in alignment:
                for fact in sent:
                    for a in fact:
                        cluster_labels[a] = cluster_lable
                    cluster_lable += 1

            scores = self_attns[eid].fill_diagonal_(0)

            for i in range(min(scores.size(0), nsent[eid])):
                for j in range(min(scores.size(1), nsent[eid])):
                    if i == j:
                        continue
                    pred_score.append(float(scores[i][j]))
                    if cluster_labels[i] == cluster_labels[j]:
                        align_label.append(1)
                    else:
                        align_label.append(0)

            pred_scores.append(pred_score)
            align_labels.append(align_label)  

        return pred_scores, align_labels


def edge_ranking_evaluation(pred_scores, gold_lables, nsents, data_path=None):

    if data_path is not None:
        gold_lables = []; pred_scores = []; nsents = []
        for line in open(data_path):
            json_obj = json.loads(line.strip())
            pred = json_obj['Pred']
            label = json_obj['Label']
            nsent = json_obj['nSent']
            if len(label) <= 1 or len(pred) <= 1:
                continue
            gold_lables.append(label)
            pred_scores.append(pred)
            nsents.append(nsent)

    scores = []; detailed_scores = {}
    for i, y_true in enumerate(gold_lables):
        y_true = np.array([y_true])
        y_score = np.array([pred_scores[i]])
        score = label_ranking_loss(y_true, y_score)
        scores.append(score)

        nsent = nsents[i]
        if nsent not in detailed_scores:
            detailed_scores[nsent] = []
        detailed_scores[nsent].append(score)

    rank_loss = sum(scores)/len(scores)
    print ('edge rank loss is:', rank_loss)

    for item in sorted(detailed_scores.items(), key = lambda d:d[0]):
        nsent = item[0]
        scores = item[1]
        rank_loss = sum(scores)/len(scores)
        print ('edge rank loss for examples with %d sentences as input is: %f' % (nsent, rank_loss))

if __name__ == '__main__':
    edge_ranking_evaluation(None, None, None, data_path='outputs.webnlg/logs.base/test.res.20000.edge')
