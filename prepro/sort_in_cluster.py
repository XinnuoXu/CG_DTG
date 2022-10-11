#coding=utf8

import json
import copy
import numpy as np
import collections
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

class SortSentsInCluster():
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


    def sort_sentences(self, sentences):
        embeddings = self.process_sentence_embedding(sentences)
        central_embedding = np.mean(embeddings, axis=0).reshape((1, -1))
        distances = pairwise_distances(embeddings, central_embedding, metric=self.metric).reshape((-1)).tolist()
        zipped = list(zip(distances, sentences))
        sorted_sentences = [item[1] for item in sorted(zipped, key=lambda d:d[0])]
        return sorted_sentences


if __name__ == '__main__':
    sort_obj = SortSentsInCluster()
    sentences = ['I love it!', 'nice product', 'good product', 'bad product', 'birds']
    sorted_sentences = sort_obj.sort_sentences(sentences)
    print (sorted_sentences)

