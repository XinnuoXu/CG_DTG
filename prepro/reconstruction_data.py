#coding=utf8

import json
import copy
import numpy as np
import collections
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

class ReconstructionData():
    def __init__(self, sentence_embedding_model='all-mpnet-base-v2',
                       device='cuda',
                       metric='euclidean',
                       compression_dimention=56,
                       semantic_threshold=0.7):

        self.device = device
        self.metric = metric
        self.semantic_threshold = semantic_threshold
        self.compression_dimention = compression_dimention

        self.s_embedding = SentenceTransformer(sentence_embedding_model, device=device)
        self.dimention_reducer = PCA(n_components=compression_dimention)


    def process_sentence_embedding(self, sentences):
        embedding = self.s_embedding.encode(sentences)
        X = np.array(embedding)
        if len(sentences) > self.compression_dimention:
            X = self.dimention_reducer.fit_transform(X)
        return X


    def select_central_sentence(self, embeddings, sentences):
        distances = pairwise_distances(embeddings, embeddings, metric=self.metric)
        avg_dist = np.mean(distances, axis=0)
        central_sentence_id = np.argmin(avg_dist)
        distances[central_sentence_id][central_sentence_id] = 1000
        closest_member = np.min(distances[central_sentence_id])
        if closest_member < self.semantic_threshold:
            new_cluster = [sentences[i] for i in range(len(sentences)) if i != central_sentence_id]
            return new_cluster, sentences[central_sentence_id]
        return sentences, ''


    def get_reconstruction_data(self, src_clusters, tgt_clusters):

        new_tgt_cluster = copy.deepcopy(tgt_clusters)
        new_src_cluster = copy.deepcopy(src_clusters)
        for key in src_clusters:
            if key == '-1':
                continue
            if key in tgt_clusters:
                continue
            cluster = src_clusters[key]
            
            sentence_embeddings = self.process_sentence_embedding(cluster)
            new_cluster, central_sentence = self.select_central_sentence(sentence_embeddings, cluster)
            if central_sentence != '':
                new_tgt_cluster[key] = ['summary of the input : '+central_sentence]
                new_src_cluster[key] = new_cluster

        return new_src_cluster, new_tgt_cluster


    def run(self, filename_in, filename_out):
        fpout = open(filename_out, 'w')
        for line in open(filename_in):
            json_obj = json.loads(line.strip())
            src_clusters = json_obj['src_clusters']
            tgt_clusters = json_obj['tgt_clusters']
            example_id = json_obj['example_id']
            raw_tgt = json_obj['raw_tgt']
            augment_src_cluster, augment_tgt_clusters = self.get_reconstruction_data(src_clusters, tgt_clusters)
            json_obj['tgt_clusters'] = augment_tgt_clusters
            json_obj['src_clusters'] = augment_src_cluster
            fpout.write(json.dumps(json_obj)+'\n')
        fpout.close()


if __name__ == '__main__':
    
    rec_obj = ReconstructionData(sentence_embedding_model='all-mpnet-base-v2',
                                 device='cuda',
                                 metric='euclidean',
                                 compression_dimention=56,
                                 semantic_threshold=0.75)

    filename_in = './outputs.ama/hdbscan_output/train.0.json'
    filename_out = './temp/train.jsonl'
    rec_obj.run(filename_in, filename_out)

