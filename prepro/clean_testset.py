#coding=utf8

import json
import copy
import numpy as np
import collections
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from prepro.review_summary_preprocess import preprocess_reviews, preprocess_summaries
#from review_summary_preprocess import preprocess_reviews, preprocess_summaries

class TgtCleaner():
    # DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    # SentenceEmb: https://www.sbert.net/examples/applications/computing-embeddings/README.html#storing-loading-embeddings
    # HDBSCAN: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    def __init__(self, sentence_embedding_model='all-mpnet-base-v2',
                       device='cuda',
                       distance_metric='euclidean',
                       distance_input_dimention=56,
                       distance_threshold=0.64):

        self.device = device

        self.distance_metric = distance_metric
        self.distance_input_dimention = distance_input_dimention
        self.distance_threshold = distance_threshold

        self.s_embedding = SentenceTransformer(sentence_embedding_model, device=device)
        self.dimention_reducer = PCA(n_components=distance_input_dimention)


    def process_sentence_embedding(self, sentences):
        embedding = self.s_embedding.encode(sentences)
        X = np.array(embedding)
        if len(sentences) > self.distance_input_dimention:
            X = self.dimention_reducer.fit_transform(X)
        return X


    def filter_target_sentences(self, targets_embeddings, source_embeddings, sources=None, targets=None):
        distances = pairwise_distances(source_embeddings, targets_embeddings, metric=self.distance_metric)
        support_relations = distances<self.distance_threshold
        support_or_not = support_relations.sum(axis=0)
        selection_labels = (support_or_not > 0)
        if sources is not None:
            for i in range(support_relations.shape[1]):
                print ('[TARGET]', targets[i])
                for j in range(support_relations.shape[0]):
                    if support_relations[j][i]:
                        print ('[SUPPORT_SRC]', sources[j], distances[j][i])
                print ('\n------------------------\n')
            print ('\n\n\n')
        return selection_labels


    def process_run(self, json_obj):

        # get sentences
        sources = preprocess_reviews(json_obj['document_segs'])
        targets, targets_prefixes = preprocess_summaries(json_obj['gold_segs'], do_sentence_segmentation=False)
        raw_tgt = json_obj['raw_tgt']
        example_id = json_obj['example_id']
        sentences = sources + targets

        # get sentence embeddings
        sentence_embeddings = self.process_sentence_embedding(sentences)
        source_embeddings = sentence_embeddings[:len(sources)]
        target_embeddings = sentence_embeddings[len(sources):]

        # classify target sentences
        #target_labels = self.filter_target_sentences(target_embeddings, source_embeddings, sources=sources, targets=targets)
        target_labels = self.filter_target_sentences(target_embeddings, source_embeddings)
        selected_sentences = [(targets_prefixes[i]+' '+targets[i]) for i in range(target_labels.shape[0]) if target_labels[i]]

        return selected_sentences


    def run(self, filename_in, filename_out):
        fpout = open(filename_out, 'w')
        for line in open(filename_in):
            json_obj = json.loads(line.strip())
            selected_tgt = self.process_run(json_obj)
            json_obj['cleaned_tgt'] = selected_tgt
            fpout.write(json.dumps(json_obj)+'\n')
        fpout.close()


if __name__ == '__main__':
    dbscan_obj = TgtCleaner(distance_metric='euclidean',
                              sentence_embedding_model='all-MiniLM-L12-v2',
                              device='cuda',
                              distance_input_dimention=56,
                              distance_threshold=0.73)
    filename_in = '/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl'
    filename_out = './temp/test.jsonl'
    dbscan_obj.run(filename_in, filename_out)

