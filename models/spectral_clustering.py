#coding=utf8

import argparse
import json
from sklearn.cluster import SpectralClustering
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SpectralCluser():
    def __init__(self, assign_labels='discretize',
                       eigen_solver='arpack',
                       affinity='precomputed',
                       max_group_size=3,
                       min_pair_freq=20,
                       train_file='', 
                       valid_file='',
                       test_file=''):

        self.assign_labels = assign_labels
        self.eigen_solver = eigen_solver
        self.affinity = affinity
        self.max_group_size = max_group_size
        self.min_pair_freq = min_pair_freq

        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file

        self.model = self.train(self.train_file)


    def train(self, train_file):
        edge_weights = {}
        for line in open(train_file):
            json_obj = json.loads(line.strip())
            preds_one_sentence = json_obj['predicates']
            for i, pred_1 in enumerate(preds_one_sentence):
                for j, pred_2 in enumerate(preds_one_sentence):
                    if i == j and len(preds_one_sentence) > 1:
                        continue
                    if pred_1 not in edge_weights:
                        edge_weights[pred_1] = {}
                    if pred_2 not in edge_weights[pred_1]:
                        edge_weights[pred_1][pred_2] = 0
                    edge_weights[pred_1][pred_2] += 1
        return edge_weights


    def process(self, predicates, n_clusters):
        ajacency_matrix = np.zeros((len(predicates),len(predicates)))
        for i, head_1 in enumerate(predicates):
            for j, head_2 in enumerate(predicates):
                if i == j:
                    continue
                if head_1 not in self.model:
                    continue
                if head_2 not in self.model[head_1]:
                    continue
                ajacency_matrix[i][j] = self.model[head_1][head_2]
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels=self.assign_labels,
                                        eigen_solver=self.eigen_solver,
                                        affinity=self.affinity).fit(ajacency_matrix)
        return clustering.labels_, ajacency_matrix


    def postprocess(self, candidates):
        candidate_map = {}
        for candidate in candidates:
            good_candidate = True
            min_freq = 100000
            for group in candidate:
                if len(group) > self.max_group_size:
                    good_candidate = False
                    break
                if len(group) == 1:
                    if (group[0] in self.model) and (group[0] in self.model[group[0]]):
                        freq = self.model[group[0]][group[0]]
                    else:
                        freq = 0
                else:
                    for i, pred_1 in enumerate(group):
                        for j, pred_2 in enumerate(group):
                            if i >= j:
                                continue
                            if pred_1 not in self.model:
                                freq = 0
                            elif pred_2 not in self.model[pred_1]:
                                freq = 0
                            else:
                                freq = self.model[pred_1][pred_2]
                if freq < min_freq:
                    min_freq = freq
                if min_freq < self.min_pair_freq:
                    good_candidate = False
                    break
            if good_candidate:
                return candidate
            if (min_freq not in candidate_map) and min_freq < 10000:
                candidate_map[min_freq] = candidate
        return sorted(candidate_map.items(), key = lambda d:d[0], reverse=True)[0][1]


    def test(self, predicates):
        candidates = []; ajacency_matrix=None
        for n_clusters in range(2, len(predicates)+1):
            labels, ajacency_matrix = self.process(predicates, n_clusters)
            pred_groups = [[] for i in range(n_clusters)]
            for i, label in enumerate(labels):
                pred_groups[label].append(predicates[i])
            candidates.append(pred_groups)
        if len(candidates) == 0:
            best_candidate = [predicates]
        else:
            best_candidate = self.postprocess(candidates)
        return best_candidate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-assign_labels", default='discretize', type=str)
    parser.add_argument("-eigen_solver", default='arpack', type=str)
    parser.add_argument("-affinity", default='precomputed', type=str)
    parser.add_argument("-max_group_size", default=3, type=int)
    parser.add_argument("-min_pair_freq", default=20, type=int)
    parser.add_argument("-train_file", default='../Plan_while_Generate/D2T_data/webnlg_partial_src/train.jsonl', type=str)
    parser.add_argument("-valid_file", default='../Plan_while_Generate/D2T_data/webnlg_data/validation.jsonl', type=str)
    parser.add_argument("-test_file", default='../Plan_while_Generate/D2T_data/webnlg_partial_src/test.jsonl', type=str)
    args = parser.parse_args()

    cluster_obj = SpectralCluser(args.assign_labels,
                                 args.eigen_solver,
                                 args.affinity,
                                 args.max_group_size,
                                 args.min_pair_freq,
                                 args.train_file)

    for line in open(args.valid_file):
        line = line.strip()
        json_obj = json.loads(line)
        predicates = json_obj['predicates']
        labels = cluster_obj.test(predicates)
