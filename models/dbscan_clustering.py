#coding=utf8

import json
import numpy as np
import collections
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

class DBSCANCluser():
    # DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    # SentenceEmb: https://www.sbert.net/examples/applications/computing-embeddings/README.html#storing-loading-embeddings
    # HDBSCAN: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
    def __init__(self, sentence_embedding_model='all-mpnet-base-v2',
                       device='cuda',
                       db_metric='euclidean',
                       db_eps=0.65,
                       db_min_samples=3,
                       db_input_dimention=64,
                       train_file='',
                       valid_file='',
                       test_file='',
                       high_freq_reviews=''):

        self.device = device
        self.db_metric = db_metric
        self.db_input_dimention = db_input_dimention

        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.high_freq_reviews = set([line.strip('\n').split('\t')[0] for line in open(high_freq_reviews)])

        self.s_embedding = SentenceTransformer(sentence_embedding_model, device=device)
        self.dimention_reducer = PCA(n_components=db_input_dimention)
        #self.clustering = DBSCAN(eps=db_eps, min_samples=db_min_samples, metric=db_metric)
        self.clustering = hdbscan.HDBSCAN(min_cluster_size=db_min_samples, min_samples=1, cluster_selection_epsilon=db_eps)

    def clean_reviews(self, sentences):

        def ugly_sentence_segmentation(sentence):
            phrases = []
            segs = sentence.split(',')
            for phrase in segs:
                smaller_granularity = phrase.split(' and ')
                for item in smaller_granularity:
                    item = item.strip()
                    phrases.append(item)
            new_phrases = []
            for i, phrase in enumerate(phrases):
                if i > 0 and len(phrase.split()) < 3:
                    new_phrases[-1] = new_phrases[-1] + ' ' + phrase
                else:
                    new_phrases.append(phrase)
            if len(new_phrases[0].split()) < 3 and len(new_phrases) > 1:
                first_phrase = new_phrases[0]
                new_phrases = new_phrases[1:]
                new_phrases[0] = first_phrase + ' ' + new_phrases[0]
            return new_phrases

        new_sentences = []
        for sentence in sentences:
            search_key = sentence.strip().lower()
            flist = search_key.split()
            search_key = ' '.join([tok for tok in flist if len(tok) > 2])
            if search_key in self.high_freq_reviews:
                continue
            sentence = sentence.lower()
            phrases = ugly_sentence_segmentation(sentence)
            phrases = [item for item in phrases if len(item.strip().split()) > 2]
            new_sentences.extend(phrases)
        return new_sentences


    def read_cluster(self, cluster_labels, sentences, target_start):
        target_sentences = ['[TARGET] '+sent for sent in sentences[target_start:]]
        sentences = sentences[:target_start] + target_sentences

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
        clustered_targets = 0
        number_of_targets = len(target_sentences)
        for i, label in enumerate(cluster_labels):
            if label not in sentence_in_clusters:
                sentence_in_clusters[label] = []
            sentence_in_clusters[label].append(sentences[i])
            if label > -1 and sentences[i].startswith('[TARGET]'):
                clustered_targets += 1

        return num_clusters, clustered_ratio, min_size, max_size, avg_size, clustered_targets, number_of_targets, sentence_in_clusters


    def print_out(self, args, example_id):

        fpout = open('temp/example.'+str(example_id), 'w')
        num_clusters, clustered_ratio, min_size, max_size, avg_size, clustered_targets, number_of_targets, sentence_in_clusters = args
        fpout.write("numer of clusters is:" + str(num_clusters) + '\n')
        fpout.write("clustered ratio is:" + str(clustered_ratio) + '\n')
        fpout.write("the size of the smallist cluster is:" + str(min_size) + '\n')
        fpout.write("the size of the largest cluster is:" + str(max_size) + '\n')
        fpout.write("argerage size of the cluaster is:" + str(avg_size) + '\n')
        fpout.write("the number of target sentences is:" + str(number_of_targets) + '\n')
        fpout.write("clustered target sentences:" + str(clustered_targets) + '\n')
        fpout.write('\n\n')

        for cluster_id in sentence_in_clusters:
            if cluster_id == -1:
                continue
            for item in sentence_in_clusters[cluster_id]:
                fpout.write(item + '\n')
            fpout.write('\n\n')
        fpout.write('######### Not in clusteres ########\n')
        for item in sentence_in_clusters[-1]:
            fpout.write(item + '\n')
        fpout.write('\n\n')
        fpout.close()


    def process_one_example(self, sentences):
        embedding = self.s_embedding.encode(sentences)
        X = np.array(embedding)
        X = self.dimention_reducer.fit_transform(X)
        ##distances = pairwise_distances(X, metric=self.db_metric)
        ##self.clustering.fit(distances)
        self.clustering.fit(X)
        return self.clustering.labels_.tolist()


    def process_train(self,):
        example_id = 0
        for line in open(self.train_file):
            json_obj = json.loads(line.strip())
            sources = self.clean_reviews(json_obj['document_segs'])
            targets = self.clean_reviews(json_obj['gold_segs'])
            sentences = sources + targets
            cluster_labels = self.process_one_example(sentences)
            args = self.read_cluster(cluster_labels, sentences, len(sources))
            self.print_out(args, example_id)
            example_id += 1
            if example_id == 15:
                break

if __name__ == '__main__':
    dbscan_obj = DBSCANCluser(db_metric='cosine',
                            sentence_embedding_model='all-MiniLM-L12-v2',
                            db_eps=0.4,
                            db_min_samples=3,
                            db_input_dimention=50,
                            train_file='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/train.jsonl',
                            valid_file='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/validation.jsonl',
                            test_file='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl',
                            high_freq_reviews='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/common_review.txt')
    dbscan_obj.process_train()

