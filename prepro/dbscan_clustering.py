#coding=utf8

import json
import copy
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
                       db_eps=0.4,
                       db_min_samples=1,
                       db_cluster_size=5,
                       db_input_dimention=56,
                       db_noise_reprocess_similar_topk=3,
                       db_noise_reprocess_threshold=0.64,
                       db_targets_similar_topk=0.2,
                       db_targets_threshold=0.8,
                       high_freq_reviews=''):

        self.device = device

        self.high_freq_reviews = None
        if high_freq_reviews != '':
            self.high_freq_reviews = set([line.strip('\n').split('\t')[0] for line in open(high_freq_reviews)])

        self.db_metric = db_metric
        self.db_noise_reprocess_similar_topk = db_noise_reprocess_similar_topk
        self.db_noise_reprocess_threshold = db_noise_reprocess_threshold
        self.db_targets_similar_topk = db_targets_similar_topk
        self.db_targets_threshold = db_targets_threshold
        self.db_input_dimention = db_input_dimention

        self.s_embedding = SentenceTransformer(sentence_embedding_model, device=device)
        self.dimention_reducer = PCA(n_components=db_input_dimention)
        self.clustering = hdbscan.HDBSCAN(min_cluster_size=db_cluster_size,
                                          min_samples=db_min_samples, 
                                          cluster_selection_epsilon=db_eps, 
                                          metric='precomputed')
        #self.clustering = DBSCAN(eps=db_eps, min_samples=db_cluster_size, metric=db_metric)


    def _cluster_statistics(self, cluster_labels):

        clusters = [item for item in cluster_labels if item != -1]
        if len(clusters) == 0:
            return 0, 0.0, 0, 0, 0.0

        counter = collections.Counter(clusters)
        min_size = min(counter.values())
        max_size = max(counter.values())
        avg_size = sum(counter.values())/len(counter)
        clustered_ratio = len(clusters)/len(cluster_labels)
        num_clusters = len(counter)

        return num_clusters, clustered_ratio, min_size, max_size, avg_size


    def _print_out(self, src_clusters, tgt_clusters, reprocessed_cluster_labels, target_labels, example_id):

        fpout = open('temp/example.'+str(example_id), 'w')

        num_clusters, clustered_ratio, min_size, max_size, avg_size = self._cluster_statistics(reprocessed_cluster_labels)
        fpout.write("numer of clusters is:" + str(num_clusters) + '\n')
        fpout.write("clustered ratio is:" + str(clustered_ratio) + '\n')
        fpout.write("the size of the smallist cluster is:" + str(min_size) + '\n')
        fpout.write("the size of the largest cluster is:" + str(max_size) + '\n')
        fpout.write("argerage size of the cluaster is:" + str(avg_size) + '\n')
        fpout.write('\n\n')

        num_clusters = max(src_clusters.keys())
        clustered_targets = 0

        for cluster_id in range(0, num_clusters+1):
            for item in src_clusters[cluster_id]:
                fpout.write(item + '\n')
            if cluster_id in tgt_clusters:
                clustered_targets += len(tgt_clusters[cluster_id])
                for item in tgt_clusters[cluster_id]:
                    fpout.write('[TARGET] ' + item + '\n')
            fpout.write('\n\n')

        fpout.write('######### Not in clusteres ########\n')
        if -1 in src_clusters:
            for item in src_clusters[-1]:
                fpout.write(item + '\n')
        if -1 in tgt_clusters:
            for item in tgt_clusters[-1]:
                fpout.write('[TARGET] ' + item + '\n')
        fpout.write('\n\n')

        fpout.write("the number of target sentences is:" + str(len(target_labels)) + '\n')
        fpout.write("clustered target sentences:" + str(clustered_targets) + '\n')

        fpout.close()


    def read_clusters(self, cluster_labels, sentences):
        sentence_in_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in sentence_in_clusters:
                sentence_in_clusters[label] = []
            sentence = sentences[i]
            sentence_in_clusters[label].append(sentence)
        return sentence_in_clusters


    def classify_sentences_into_clusters(self, cluster_labels, new_cluster_labels, distances, similar_topk, threshold):
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(distances[i])
        for label in clusters:
            clusters[label] = np.stack(clusters[label], axis=0)

        classification_scores = [-1] * len(new_cluster_labels)
        for i, label in enumerate(new_cluster_labels):
            if label != -1:
                continue
            closest_cluster_id = -1; closest_cluster_distance = 1000.0
            for key in clusters:
                one_to_many_distances = clusters[key][:, i]
                if similar_topk < 1:
                    similar_topk = max(3, int(similar_topk * one_to_many_distances.shape[-1]))
                top_k = min(one_to_many_distances.shape[-1], similar_topk)
                top_k_distances = np.sort(one_to_many_distances)[:top_k]
                mean_distance = np.mean(top_k_distances)
                if mean_distance < closest_cluster_distance:
                    closest_cluster_id = key
                    closest_cluster_distance = mean_distance
            if closest_cluster_id > -1 and closest_cluster_distance < threshold:
                new_cluster_labels[i] = closest_cluster_id
                classification_scores[i] = closest_cluster_distance
        return new_cluster_labels, classification_scores


    def process_clustering(self, sentence_embeddings):
        distances = pairwise_distances(sentence_embeddings, metric=self.db_metric)
        self.clustering.fit(distances.astype(np.double))
        ##self.clustering.fit(sentence_embeddings)
        return self.clustering.labels_.tolist(), distances


    def process_sentence_embedding(self, sentences):
        embedding = self.s_embedding.encode(sentences)
        X = np.array(embedding)
        if len(sentences) > self.db_input_dimention:
            X = self.dimention_reducer.fit_transform(X)
        return X


    def classify_target_sentences(self, targets_embeddings, source_embeddings, cluster_labels):
        distances = pairwise_distances(source_embeddings, targets_embeddings, metric=self.db_metric)
        target_sentence_labels = [-1] * targets_embeddings.shape[0]
        target_sentence_labels, classification_scores = self.classify_sentences_into_clusters(cluster_labels, 
                                                                       target_sentence_labels, distances,
                                                                       self.db_targets_similar_topk,
                                                                       self.db_targets_threshold)
        return target_sentence_labels, classification_scores


    def process_run(self, line):
        json_obj = json.loads(line.strip())
            
        # get sentences
        sources = preprocess_reviews(json_obj['document_segs'], high_freq_reviews=self.high_freq_reviews)
        targets, targets_prefixes = preprocess_summaries(json_obj['gold_segs'])
        raw_tgt = json_obj['raw_tgt']
        example_id = json_obj['example_id']
        sentences = sources + targets

        # get sentence embeddings
        sentence_embeddings = self.process_sentence_embedding(sentences)
        source_embeddings = sentence_embeddings[:len(sources)]
        target_embeddings = sentence_embeddings[len(sources):]

        # cluster input sentences
        cluster_labels, distances = self.process_clustering(source_embeddings)
        reprocessed_cluster_labels = copy.deepcopy(cluster_labels)
        reprocessed_cluster_labels, add_sentences_scores = self.classify_sentences_into_clusters(cluster_labels, 
                                                                               reprocessed_cluster_labels, 
                                                                               distances, 
                                                                               self.db_noise_reprocess_similar_topk, 
                                                                               self.db_noise_reprocess_threshold)

        # classify target sentences
        target_labels, target_scores = self.classify_target_sentences(target_embeddings, 
                                                                      source_embeddings, 
                                                                      reprocessed_cluster_labels)

        # read clusters
        src_clusters = self.read_clusters(reprocessed_cluster_labels, sources)
        for i, text in enumerate(targets):
            targets[i] = targets_prefixes[i] + ' ' + targets[i]
        tgt_clusters = self.read_clusters(target_labels, targets)

        #self._print_out(src_clusters, tgt_clusters, reprocessed_cluster_labels, target_labels, example_id)

        return src_clusters, tgt_clusters, example_id, raw_tgt


    def run(self, filename_in, filename_out):
        fpout = open(filename_out, 'w')
        for line in open(filename_in):
            src_clusters, tgt_clusters, example_id, raw_tgt = self.process_run(line.strip())
            output_json = {}
            output_json['src_clusters'] = src_clusters
            output_json['tgt_clusters'] = tgt_clusters
            output_json['example_id'] = example_id
            output_json['raw_tgt'] = raw_tgt
            fpout.write(json.dumps(output_json)+'\n')
        fpout.close()


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


def preprocess_reviews(sentences, high_freq_reviews=None):
    new_sentences = []
    for sentence in sentences:
        search_key = sentence.strip().lower()
        flist = search_key.split()
        search_key = ' '.join([tok for tok in flist if len(tok) > 2])
        if high_freq_reviews != None and search_key in high_freq_reviews:
            continue
        sentence = sentence.lower()
        phrases = ugly_sentence_segmentation(sentence)
        phrases = [item for item in phrases if len(item.strip().split()) > 2]
        new_sentences.extend(phrases)
    return new_sentences


def preprocess_summaries(sentences):
    new_sentences = []; new_prefix = []
    for sentence in sentences:
        sentence = sentence.lower()
        prefix = ' '.join(sentence.split()[:5])
        sentence = ' '.join(sentence.split()[5:])
        phrases = ugly_sentence_segmentation(sentence)
        phrases = [item for item in phrases]
        prefixes = [prefix for item in phrases]
        new_sentences.extend(phrases)
        new_prefix.extend(prefixes)
    return new_sentences, new_prefix


if __name__ == '__main__':
    dbscan_obj = DBSCANCluser(db_metric='euclidean',
                              sentence_embedding_model='all-MiniLM-L12-v2',
                              db_eps=0.4,
                              db_cluster_size=5,
                              db_input_dimention=56,
                              db_noise_reprocess_similar_topk=3,
                              db_noise_reprocess_threshold=0.6,
                              db_targets_similar_topk=0.2,
                              db_targets_threshold=0.8,                              
                              high_freq_reviews='/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/common_review.txt')
    filename_in = '/home/hpcxu1/Planning/Plan_while_Generate/AmaSum/AmaSum_data/train.jsonl'
    filename_out = './temp/train.jsonl'
    dbscan_obj.run(filename_in, filename_out)

