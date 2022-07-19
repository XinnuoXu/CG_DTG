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
    def __init__(self, args):
        self.assign_labels = args.assign_labels
        self.eigen_solver = args.eigen_solver
        self.affinity = args.affinity
        self.train_file = args.train_file
        self.valid_file = args.valid_file
        self.test_file = args.test_file
        self.model = self.train(self.train_file)

    def train(self, train_file):
        edge_weights = {}
        for line in open(train_file):
            json_obj = json.loads(line.strip())
            preds_one_sentence = json_obj['predicates']
            for i, pred_1 in enumerate(preds_one_sentence):
                for j, pred_2 in enumerate(preds_one_sentence):
                    if i == j:
                        continue
                    if pred_1 not in edge_weights:
                        edge_weights[pred_1] = {}
                    if pred_2 not in edge_weights[pred_1]:
                        edge_weights[pred_1][pred_2] = 0
                    edge_weights[pred_1][pred_2] += 1
        return edge_weights

    def test(self, predicates, n_clusters):
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
        print (ajacency_matrix)
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels=self.assign_labels,
                                        eigen_solver=self.eigen_solver,
                                        affinity=self.affinity).fit(ajacency_matrix)
        return clustering.labels_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-assign_labels", default='discretize', type=str)
    parser.add_argument("-eigen_solver", default='arpack', type=str)
    parser.add_argument("-affinity", default='precomputed', type=str)
    parser.add_argument("-train_file", default='../Plan_while_Generate/D2T_data/webnlg_partial_src/train.jsonl', type=str)
    parser.add_argument("-valid_file", default='../Plan_while_Generate/D2T_data/webnlg_data/validation.jsonl', type=str)
    parser.add_argument("-test_file", default='../Plan_while_Generate/D2T_data/webnlg_partial_src/test.jsonl', type=str)
    args = parser.parse_args()

    cluster_obj = SpectralCluser(args)
    for line in open(args.valid_file):
        line = line.strip()
        json_obj = json.loads(line)
        predicates = json_obj['predicates']
        for n_clusters in range(2, len(predicates)):
            labels = cluster_obj.test(predicates, n_clusters)
            print (n_clusters, predicates, labels)
