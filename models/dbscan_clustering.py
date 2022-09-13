#coding=utf8

import json
import numpy as np
import collections
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

class DBSCANCluser():
    # DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    # SentenceEmb: https://www.sbert.net/examples/applications/computing-embeddings/README.html#storing-loading-embeddings
    def __init__(self, sentence_embedding_model='all-mpnet-base-v2',
                       device='cuda',
                       db_metric='euclidean',
                       db_eps=0.65,
                       db_min_samples=3,
                       train_file='',
                       valid_file='',
                       test_file='',
                       high_freq_reviews=''):

        self.s_embedding = SentenceTransformer(sentence_embedding_model)
        #self.s_embedding = SentenceTransformer(sentence_embedding_model, device=device)
        self.clustering = DBSCAN(eps=db_eps, min_samples=db_min_samples, metric='precomputed')
        self.device = device
        self.db_metric = db_metric
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.high_freq_reviews = set([line.strip('\n').split('\t')[0] for line in open(high_freq_reviews)])

    def clean_reviews(self, sentences):
        new_sentences = []
        for sentence in sentences:
            search_key = sentence.strip().lower()
            flist = search_key.split()
            search_key = ' '.join([tok for tok in flist if len(tok) > 2])
            if search_key in self.high_freq_reviews:
                continue
            new_sentences.append(sentence.lower())
        return new_sentences

    def process_one_example(self, sentences):
        embedding = self.s_embedding.encode(sentences)
        X = np.array(embedding)
        distances = pairwise_distances(X, metric=self.db_metric)
        self.clustering.fit(distances)
        return self.clustering.labels_.tolist()

    def read_cluster(self, cluster_labels, sentences):
        clusters = [item for item in cluster_labels if item != -1]
        if len(clusters) == 0:
            return 0, 0.0, 0, 0, 0.0
        counter = collections.Counter(clusters)
        min_size = min(counter.values())
        max_size = max(counter.values())
        avg_size = sum(counter.values())/len(counter)
        clustered_ratio = len(clusters)/len(cluster_labels)
        num_clusters = len(counter)
        sentence_in_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in sentence_in_clusters:
                sentence_in_clusters[label] = []
            sentence_in_clusters[label].append(sentences[i])
        for cluster_id in sentence_in_clusters:
            if cluster_id == -1:
                continue
            for item in sentence_in_clusters[cluster_id]:
                print (item)
            print ('\n\n')
        print ('######### Not in clusteres ########')
        for item in sentence_in_clusters[-1]:
            print (item)
        print ('\n\n')
        return num_clusters, clustered_ratio, min_size, max_size, avg_size

    def process_train(self,):
        for line in open(self.train_file):
            json_obj = json.loads(line.strip())
            sources = self.clean_reviews(json_obj['document_segs'])
            targets = json_obj['gold_segs']
            sentences = sources + targets
            cluster_labels = self.process_one_example(sentences)
            num_clusters, clustered_ratio, min_size, max_size, avg_size = self.read_cluster(cluster_labels)
            print (num_clusters, clustered_ratio, min_size, max_size, avg_size)

if __name__ == '__main__':
    dbscan_obj = DBSCANCluser(db_metric='cosine',
                            db_eps=0.4,
                            train_file='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/train.jsonl',
                            valid_file='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/validation.jsonl',
                            test_file='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl',
                            high_freq_reviews='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/common_review.txt')
    dbscan_obj.process_train()

