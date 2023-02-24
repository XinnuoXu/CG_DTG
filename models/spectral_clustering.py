#coding=utf8

import argparse
import json, random
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
    def __init__(self, method='spectral_clustering',
                       assign_labels='discretize',
                       eigen_solver='arpack',
                       affinity='precomputed',
                       pair_frequency_threshold=0.0,
                       max_group_size=3,
                       min_pair_freq=20,
                       use_ratio=False,
                       filter_with_entities=True,
                       train_file='', 
                       valid_file='',
                       test_file=''):

        self.FIRST_SENT_LABEL = '<FIRST_SENT>'
        self.method = method
        self.assign_labels = assign_labels
        self.eigen_solver = eigen_solver
        self.affinity = affinity
        self.pair_frequency_threshold = pair_frequency_threshold
        self.max_group_size = max_group_size
        self.min_pair_freq = min_pair_freq
        self.use_ratio = use_ratio
        self.filter_with_entities = filter_with_entities

        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file

        self.model = self.train(self.train_file)
        self.model_ratio = self.train_freq_to_ration(self.model)

        self.max_ngroup = 10
        self.nsent_dist = self.nsent_distribution(self.train_file, self.max_ngroup)


    def one_sentence(self, edge_weights, preds_one_sentence):
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

    def train(self, train_file):
        edge_weights = {}
        for line in open(train_file):
            json_obj = json.loads(line.strip())
            alignments = json_obj['oracles_selection']
            #if len(alignments) == 1:
            #    continue
            predicates = json_obj['predicates']
            if len(predicates) == 1:
                continue
            groups = []
            for idx_group in alignments:
                groups.append([predicates[idx] for idx in idx_group])
            for i, group in enumerate(groups):
                #group = group.split(' ')
                if i == 0 and len(groups) > 1:
                    group.append(self.FIRST_SENT_LABEL)
                edge_weights = self.one_sentence(edge_weights, group)

        for key1 in edge_weights:
            for key2 in edge_weights[key1]:
                edge_weights[key1][key2] = edge_weights[key1][key2]

        return edge_weights

    def train_freq_to_ration(self, edge_weights):
        ratio_edge_weights = {}
        for pred_1 in edge_weights:
            if pred_1 == self.FIRST_SENT_LABEL:
                pred_1_freq = 0
            else:
                pred_1_freq = sum(edge_weights[pred_1].values())
            ratio_edge_weights[pred_1] = {}
            for pred_2 in edge_weights[pred_1]:
                if pred_2 == self.FIRST_SENT_LABEL:
                    pred_2_freq = 0
                else:
                    pred_2_freq = sum(edge_weights[pred_2].values())
                if pred_1_freq+pred_2_freq == 0:
                    continue
                ratio_edge_weights[pred_1][pred_2] = ((edge_weights[pred_1][pred_2])*2)/(pred_1_freq+pred_2_freq)
        return ratio_edge_weights


    def nsent_distribution(self, train_file, max_ngroup):
        ngroup_dict = {}
        for line in open(train_file):
            json_obj = json.loads(line.strip())
            #groups = json_obj['prompt_str'].split(' ||| ')
            groups = json_obj['oracles_selection']
            npred = len(json_obj['predicates'])
            if len(groups) > npred:
                continue
            if npred not in ngroup_dict:
                ngroup_dict[npred] = [0] * (npred+1)
            ngroup_dict[npred][len(groups)] += 1

        ngroup_dist = {}
        for npred in ngroup_dict:
            ngroup = ngroup_dict[npred]
            ngroup_dist[npred] = [size/sum(ngroup) for size in ngroup]
        return ngroup_dist


    def process(self, predicates, pred_to_subs_objs, n_clusters):
        ajacency_matrix = np.zeros((len(predicates),len(predicates)))
        for i, head_1 in enumerate(predicates):
            for j, head_2 in enumerate(predicates):
                if i == j:
                    continue
                if head_1 not in self.model:
                    continue
                if head_2 not in self.model[head_1]:
                    continue
                if self.use_ratio:
                    ajacency_matrix[i][j] = self.model_ratio[head_1][head_2]
                else:
                    ajacency_matrix[i][j] = self.model[head_1][head_2]
                if self.filter_with_entities and len(set(pred_to_subs_objs[head_1])&set(pred_to_subs_objs[head_2])) == 0:
                    ajacency_matrix[i][j] = 0.0
                if ajacency_matrix[i][j] < self.pair_frequency_threshold:
                    ajacency_matrix[i][j] = 0.0

        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels=self.assign_labels,
                                        eigen_solver=self.eigen_solver,
                                        affinity=self.affinity).fit(ajacency_matrix)
        return clustering.labels_, ajacency_matrix


    def select_best_candidate(self, candidates):

        candidate_map = {}
        for candidate in candidates:
            #print ('*******')
            #print (candidate)

            # if the candidate contains a very big group or empty group
            bad_group = False
            for group in candidate:
                if len(group) > self.max_group_size:
                    bad_group = True
                    break
            if bad_group:
                continue

            # if the candidate contains low frequence groups
            good_candidate = True; min_freq = 10000
            for group in candidate:
                if len(group) == 0:
                    continue
                group_min_freq = 100000
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
                        #print ('Pair frequence:', pred_1, pred_2, freq)
                        if freq < group_min_freq:
                            group_min_freq = freq
                #print ('Group lowest frequence:', group, group_min_freq)
                if group_min_freq < self.min_pair_freq:
                    good_candidate = False
                if group_min_freq < min_freq:
                    min_freq = group_min_freq
            if good_candidate:
                return candidate

            if (min_freq not in candidate_map) and min_freq < 100000:
                candidate_map[min_freq] = candidate

        return sorted(candidate_map.items(), key = lambda d:d[0], reverse=True)[0][1]


    def sort_the_groups(self, best_candidate):
        first_sent_freq = {}
        for i, group in enumerate(best_candidate):
            if len(group) == 0:
                continue
            freq_sum = 0
            for pred in group:
                if pred in self.model[self.FIRST_SENT_LABEL]:
                    first_freq = self.model[self.FIRST_SENT_LABEL][pred]
                    freq_sum += first_freq
                else:
                    freq_sum += 0
            first_sent_freq[i] = freq_sum / float(len(group)) 
        new_best_candidate = []
        for item in sorted(first_sent_freq.items(), key = lambda d:d[1], reverse=True):
            sentence_idx = item[0]
            new_best_candidate.append(best_candidate[sentence_idx])
        return new_best_candidate


    def sort_the_groups_v2(self, best_candidate, pred_to_subs_objs):
        # just for a taste, not elegant
        first_sent_freq = {}
        for i, group in enumerate(best_candidate):
            if len(group) == 0:
                continue
            freq_sum = 0
            for pred in group:
                if pred in self.model[self.FIRST_SENT_LABEL]:
                    first_freq = self.model[self.FIRST_SENT_LABEL][pred]
                    freq_sum += first_freq
                else:
                    freq_sum += 0
            first_sent_freq[i] = freq_sum / float(len(group)) 
        new_best_candidate_same_sub = []; new_best_candidate_not_same_sub = []
        main_sub = ''
        for item in sorted(first_sent_freq.items(), key = lambda d:d[1], reverse=True):
            sentence_idx = item[0]
            current_group = best_candidate[sentence_idx]
            if main_sub == '':
                subjects = [pred_to_subs_objs[pred][0] for pred in current_group]
                main_sub = max(set(subjects), key=subjects.count)
                new_best_candidate_same_sub.append(current_group)
            else:
                subjects = [pred_to_subs_objs[pred][0] for pred in current_group]
                max_sub = max(set(subjects), key=subjects.count)
                if max_sub == main_sub:
                    new_best_candidate_same_sub.append(current_group)
                else:
                    new_best_candidate_not_same_sub.append(current_group)
        return new_best_candidate_same_sub + new_best_candidate_not_same_sub


    def triples_to_entityes(self, triples, predicates):
        pred_to_subs_objs = {}
        for i, triple in enumerate(triples):
            pred = predicates[i]
            sub_idx = triple.find('<SUB>')
            obj_idx = triple.find('<OBJ>')
            pred_idx = triple.find('<PRED>')
            sub = triple[sub_idx+5:pred_idx].strip()
            obj = triple[obj_idx+5:].strip()
            pred_to_subs_objs[pred] = [sub, obj]
        return pred_to_subs_objs


    def get_aggragation_lable(self, predicates, triples, ncluster):
        if len(predicates) == 1:
            return [0]
        pred_to_subs_objs = self.triples_to_entityes(triples, predicates)
        labels, ajacency_matrix = self.process(predicates, pred_to_subs_objs, ncluster)
        return labels


    def calculate_graph_score(self, labels, predicates, n_clusters):

        pred_groups = [[] for i in range(n_clusters)]
        for i, label in enumerate(labels):
            pred_groups[label].append(predicates[i])
        candidate = pred_groups

        for group in candidate:
            if len(group) > self.max_group_size:
                return [-1 for i in range(n_clusters)]
            if len(group) == 0:
                return [-1 for i in range(n_clusters)]

        scores = []
        for group in candidate:
            group_min_freq = 100000
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
                    #print ('Pair frequence:', pred_1, pred_2, freq)
                    if freq < group_min_freq:
                        group_min_freq = freq
            scores.append(group_min_freq)

        return scores


    def run_test(self, predicates, triples, gt_ncluster=None):

        candidates = []; ajacency_matrix=None
        if len(predicates) == 1:
            return [predicates]

        pred_to_subs_objs = self.triples_to_entityes(triples, predicates)

        for n_clusters in range(1, len(predicates)+1):
            if (gt_ncluster is not None) and (n_clusters != gt_ncluster):
                continue
            labels, ajacency_matrix = self.process(predicates, pred_to_subs_objs, n_clusters)
            pred_groups = [[] for i in range(n_clusters)]
            for i, label in enumerate(labels):
                pred_groups[label].append(predicates[i])
            candidates.append(pred_groups)
        #print (ajacency_matrix)

        if len(candidates) == 0:
            best_candidate = [predicates]
        else:
            best_candidate = self.select_best_candidate(candidates)
        #sorted_best_candidate = self.sort_the_groups(best_candidate)
        sorted_best_candidate = self.sort_the_groups_v2(best_candidate, pred_to_subs_objs)

        return sorted_best_candidate


    def run_random(self, predicates, triples):
        
        n_clusters = np.random.choice(np.arange(0, len(predicates)+1), p=self.nsent_dist[len(predicates)])
        labels = [random.sample(range(n_clusters), 1)[0] for item in predicates]
        pred_groups = [[] for i in range(n_clusters)]
        for i, label in enumerate(labels):
            pred_groups[label].append(predicates[i])
        candidate = pred_groups

        pred_to_subs_objs = self.triples_to_entityes(triples, predicates)
        sorted_candidate = self.sort_the_groups_v2(candidate, pred_to_subs_objs)
        
        return sorted_candidate


    def run_entity(self, predicates, triples):

        if len(predicates) == 1:
            return [predicates]
        
        n_clusters = np.random.choice(np.arange(0, len(predicates)+1), p=self.nsent_dist[len(predicates)])
        pred_to_subs_objs = self.triples_to_entityes(triples, predicates)
        
        ajacency_matrix = np.ones((len(predicates),len(predicates))) - np.identity(len(predicates))
        for i, head_1 in enumerate(predicates):
            for j, head_2 in enumerate(predicates):
                if i == j:
                    continue
                if len(set(pred_to_subs_objs[head_1])&set(pred_to_subs_objs[head_2])) == 0:
                    ajacency_matrix[i][j] = 0.0
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels=self.assign_labels,
                                        eigen_solver=self.eigen_solver,
                                        affinity=self.affinity).fit(ajacency_matrix)
        labels = clustering.labels_

        pred_groups = [[] for i in range(n_clusters)]
        for i, label in enumerate(labels):
            pred_groups[label].append(predicates[i])
        candidate = pred_groups

        if ajacency_matrix.sum() == 0:
            candidate = [[pred] for pred in predicates]

        sorted_candidate = self.sort_the_groups_v2(candidate, pred_to_subs_objs)
        
        return sorted_candidate


    def run(self, predicates, triples, prompt_str=None, gt_ncluster=None):
        if self.method == 'random':
            candidate = self.run_random(predicates, triples)
        elif self.method == 'only_entity':
            candidate = self.run_entity(predicates, triples)
        elif self.method == 'ground_truth':
            candidate = [item.split() for item in prompt_str.split('<ref-sep>')[1].split(' ||| ')]
        else:
            candidate = self.run_test(predicates, triples, gt_ncluster=gt_ncluster)
        return candidate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", default='spectral_clustering', type=str, choices=['spectral_clustering', 'random', 'only_entity', 'ground_truth'])
    parser.add_argument("-assign_labels", default='discretize', type=str)
    parser.add_argument("-eigen_solver", default='arpack', type=str)
    parser.add_argument("-affinity", default='precomputed', type=str)
    parser.add_argument("-pair_frequency_threshold", default=0.0, type=float)
    parser.add_argument("-max_group_size", default=3, type=int)
    parser.add_argument("-min_pair_freq", default=20, type=int)
    parser.add_argument("-use_ratio", type=str2bool, default=False)
    parser.add_argument("-filter_with_entities", type=str2bool, default=True)
    parser.add_argument("-train_file", default='../Plan_while_Generate/D2T_data/webnlg_data/train.jsonl', type=str)
    parser.add_argument("-valid_file", default='../Plan_while_Generate/D2T_data/webnlg_data/validation.jsonl', type=str)
    parser.add_argument("-test_file", default='../Plan_while_Generate/D2T_data/webnlg_partial_src/test.jsonl', type=str)
    args = parser.parse_args()

    cluster_obj = SpectralCluser(args.method,
                                 args.assign_labels,
                                 args.eigen_solver,
                                 args.affinity,
                                 args.pair_frequency_threshold,
                                 args.max_group_size,
                                 args.min_pair_freq,
                                 args.use_ratio,
                                 args.filter_with_entities,
                                 args.train_file)

    '''
    for line in open(args.valid_file):
        line = line.strip()
        json_obj = json.loads(line)
        predicates = json_obj['predicates']
        labels = cluster_obj.test(predicates)
        #print ('PREDICATES', predicates)
        #print ('GROUPS', labels)
        #print ('\n')
    '''

    #predicates = ['-Pred-country', '-Pred-isPartOf', '-Pred-capital', '-Pred-state']
    #predicates = ['-Pred-country', '-Pred-isPartOf', '-Pred-capital', '-Pred-isPartOf_1']
    #predicates = ['-Pred-country', '-Pred-areaTotal', '-Pred-isPartOf', '-Pred-isPartOf_1']
    #predicates = ['-Pred-populationDensity', '-Pred-areaTotal', '-Pred-isPartOf', '-Pred-elevationAboveTheSeaLevel']
    #predicates = ['-Pred-country', '-Pred-language', '-Pred-capital', '-Pred-leaderName', '-Pred-leaderName_1']
    #predicates = ['-Pred-capital', '-Pred-country', '-Pred-demonym', '-Pred-leaderName', '-Pred-leaderName_1']
    #predicates = ['-Pred-capital', '-Pred-country', '-Pred-leaderName', '-Pred-leaderName_1']
    #predicates = ['-Pred-capital', '-Pred-country', '-Pred-ethnicGroup', '-Pred-largestCity', '-Pred-isPartOf']
    #predicates = ['-Pred-leaderName', '-Pred-country', '-Pred-cityServed', '-Pred-isPartOf', '-Pred-location']
    #predicates = ['-Pred-birthDate', '-Pred-birthPlace', '-Pred-nationality', '-Pred-operator' ,'-Pred-was_a_crew_member_of', '-Pred-was_selected_by_NASA', '-Pred-status']
    #predicates = ['-Pred-operator' ,'-Pred-was_a_crew_member_of', '-Pred-was_selected_by_NASA']
    #predicates = ['-Pred-country', '-Pred-ethnicGroup', '-Pred-isPartOf', '-Pred-state']
    #predicates = ['-Pred-country', '-Pred-ethnicGroup', '-Pred-ethnicGroup_1', '-Pred-language']
    #predicates = ['-Pred-country', '-Pred-location', '-Pred-ground', '-Pred-league']
    #predicates = ['-Pred-country', '-Pred-location', '-Pred-location_1', '-Pred-category', '-Pred-established', '-Pred-state']
    #predicates = ['-Pred-country', '-Pred-ethnicGroup', '-Pred-isPartOf', '-Pred-capital']
    #predicates = ['-Pred-architect', '-Pred-birthPlace', '-Pred-significantBuilding', '-Pred-significantBuilding_1', '-Pred-significantProject']

    #triples = ['<SUB> mason school of business <PRED> -Pred-country <OBJ> united states', '<SUB> alan b. miller hall<PRED>-Pred-location<OBJ> virginia', '<SUB> alan b. miller hall<PRED>-Pred-architect<OBJ> robert a. m. stern', '<SUB> alan b. miller hall<PRED>-Pred-owner<OBJ> college of william & mary', '<SUB> alan b. miller hall<PRED>-Pred-currentTenants<OBJ> mason school of business']
    #predicates = ['-Pred-country', '-Pred-location', '-Pred-architect', '-Pred-owner', '-Pred-currentTenants']

    #triples = ['<SUB> 11th mississippi infantry monument<PRED>-Pred-category<OBJ> contributing property', '<SUB> 11th mississippi infantry monument<PRED>-Pred-location<OBJ> adams county, pennsylvania', '<SUB> adams county, pennsylvania<PRED>-Pred-has_to_its_north<OBJ> cumberland county, pennsylvania', '<SUB> adams county, pennsylvania<PRED>-Pred-has_to_its_southeast<OBJ> carroll county, maryland', '<SUB> adams county, pennsylvania<PRED>-Pred-has_to_its_southwest<OBJ> frederick county, maryland']
    #predicates = ['-Pred-category', '-Pred-location', '-Pred-has_to_its_north', '-Pred-has_to_its_southeast', '-Pred-has_to_its_southwest']

    #triples = ['<SUB> 11th mississippi infantry monument<PRED>-Pred-country<OBJ> "united states"', '<SUB> 11th mississippi infantry monument<PRED>-Pred-location<OBJ> seminary ridge', '<SUB> 11th mississippi infantry monument<PRED>-Pred-category<OBJ> contributing property', '<SUB> 11th mississippi infantry monument<PRED>-Pred-established<OBJ> 2000', '<SUB> 11th mississippi infantry monument<PRED>-Pred-municipality<OBJ> gettysburg, pennsylvania']
    #predicates = ['-Pred-country', '-Pred-location', '-Pred-category', '-Pred-established', '-Pred-municipality']

    #triples = ['<SUB> auburn, alabama<PRED>-Pred-isPartOf<OBJ> lee county, alabama ', '<SUB> alabama<PRED>-Pred-country<OBJ> united states ', '<SUB> lee county, alabama<PRED>-Pred-state<OBJ> alabama ', '<SUB> united states<PRED>-Pred-ethnicGroup<OBJ> african americans ','<SUB> united states<PRED>-Pred-capital<OBJ> washington, d.c.']
    #predicates = ['-Pred-isPartOf', '-Pred-country', '-Pred-state', '-Pred-ethnicGroup', '-Pred-capital']

    #triples = ['<SUB> italy<PRED>-Pred-demonym<OBJ> italians ', '<SUB> italy<PRED>-Pred-capital<OBJ> rome', '<SUB> amatriciana sauce<PRED>-Pred-country<OBJ> italy', '<SUB> italy<PRED>-Pred-leaderName<OBJ> sergio mattarella', '<SUB> italy<PRED>-Pred-leaderName_1<OBJ> laura boldrini']
    #predicates = ['-Pred-demonym', '-Pred-capital', '-Pred-country', '-Pred-leaderName', '-Pred-leaderName_1']

    #triples = ['<SUB> azerbaijan<PRED>-Pred-capital<OBJ> baku ', '<SUB> baku turkish martyrs memorial<PRED>-Pred-dedicatedTo<OBJ> "ottoman army soldiers killed in the battle of baku" ', '<SUB> baku turkish martyrs memorial<PRED>-Pred-location<OBJ> azerbaijan ', '<SUB> azerbaijan<PRED>-Pred-leaderName<OBJ> artur rasizade', '<SUB> azerbaijan<PRED>-Pred-legislature<OBJ> national assembly (azerbaijan)']
    #predicates = ['-Pred-capital', '-Pred-dedicatedTo', '-Pred-location', '-Pred-leaderName', '-Pred-legislature']

    #triples = ['<SUB> united states<PRED>-Pred-capital<OBJ> washington, d.c. ', '<SUB> united states<PRED>-Pred-ethnicGroup<OBJ> white americans ', '<SUB> new jersey<PRED>-Pred-country<OBJ> united states ', '<SUB> united states<PRED>-Pred-largestCity<OBJ> new york city ', '<SUB> atlantic city, new jersey<PRED>-Pred-isPartOf<OBJ> new jersey']
    #predicates = ['-Pred-capital', '-Pred-ethnicGroup', '-Pred-country', '-Pred-largestCity', '-Pred-isPartOf']

    #triples = ['<SUB> aenir<PRED>-Pred-author<OBJ> garth nix ', '<SUB> aenir<PRED>-Pred-language<OBJ> english language ', '<SUB> aenir<PRED>-Pred-followedBy<OBJ> above the veil']
    #predicates = ['-Pred-author', '-Pred-language', '-Pred-followedBy']

    #triples = ['<SUB> gujarat<PRED>-Pred-leaderName<OBJ> anandiben patel ', '<SUB> amdavad ni gufa<PRED>-Pred-location<OBJ> gujarat ', '<SUB> amdavad ni gufa<PRED>-Pred-country<OBJ> india ', '<SUB> india<PRED>-Pred-leaderName_1<OBJ> sumitra mahajan']
    #predicates = ['-Pred-leaderName', '-Pred-location', '-Pred-country', '-Pred-leaderName_1']

    #triples = ['<SUB> alan b. miller hall<PRED>-Pred-buildingStartDate<OBJ> "30 march 2007" ', '<SUB> mason school of business<PRED>-Pred-country<OBJ> united states ', '<SUB> alan b. miller hall<PRED>-Pred-currentTenants<OBJ> mason school of business']
    #predicates = ['-Pred-buildingStartDate', '-Pred-country', '-Pred-currentTenants']

    #triples = ['<SUB> 103 colmore row<PRED>-Pred-location<OBJ> birmingham ', '<SUB> 103 colmore row<PRED>-Pred-completionDate<OBJ> 1976 ', '<SUB> 103 colmore row<PRED>-Pred-buildingStartDate<OBJ> "1973" ', '<SUB> 103 colmore row<PRED>-Pred-floorCount<OBJ> 23']
    #predicates = ['-Pred-location', '-Pred-completionDate', '-Pred-buildingStartDate', '-Pred-floorCount']

    #triples = ['<SUB> turkey<PRED>-Pred-leaderTitle<OBJ> president of turkey', '<SUB> turkey<PRED>-Pred-leaderName<OBJ> ahmet davuto<unk>lu ', '<SUB> turkey<PRED>-Pred-capital<OBJ> ankara ', '<SUB> turkey<PRED>-Pred-largestCity<OBJ> istanbul ', '<SUB> turkey<PRED>-Pred-currency<OBJ> turkish lira ', '<SUB> atatürk monument (i<unk>zmir)<PRED>-Pred-inaugurationDate<OBJ> "1932-07-27"', '<SUB> atatürk monument (i<unk>zmir)<PRED>-Pred-location<OBJ> turkey']
    #predicates = ['-Pred-leaderTitle', '-Pred-leaderName', '-Pred-capital', '-Pred-largestCity', '-Pred-currency', '-Pred-inaugurationDate', '-Pred-location']

    #triples = ['<SUB> turkey<PRED>-Pred-leaderTitle<OBJ> president of turkey', ' <SUB> turkey<PRED>-Pred-leader<OBJ> ahmet davuto<unk>lu ', '<SUB> atatürk monument (i<unk>zmir)<PRED>-Pred-designer<OBJ> pietro canonica ', '<SUB> turkey<PRED>-Pred-capital<OBJ> ankara ', '<SUB> atatürk monument (i<unk>zmir)<PRED>-Pred-material<OBJ> "bronze" ', '<SUB> atatürk monument (i<unk>zmir)<PRED>-Pred-inaugurationDate<OBJ> "1932-07-27" ', '<SUB> atatürk monument (i<unk>zmir)<PRED>-Pred-location<OBJ> turkey']
    #predicates = ['-Pred-leaderTitle', '-Pred-leader', '-Pred-designer', '-Pred-capital', '-Pred-material', '-Pred-inaugurationDate', '-Pred-location']

    #triples = ['<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-academicStaffSize<OBJ> 737 ', '<SUB> denmark<PRED>-Pred-leaderName<OBJ> lars l<unk>kke rasmussen ', '<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-dean<OBJ> "thomas pallesen" ', '<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-city<OBJ> aarhus ', '<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-country<OBJ> denmark', ' <SUB> school of business and social sciences at the aarhus university<PRED>-Pred-affiliation<OBJ> european university association ', '<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-established<OBJ> 1928']
    #predicates = ['-Pred-academicStaffSize', '-Pred-leaderName', '-Pred-dean', '-Pred-city', '-Pred-country', '-Pred-affiliation', '-Pred-established']

    #triples = ['<SUB> california<PRED>-Pred-gemstone<OBJ> benitoite ','<SUB> california<PRED>-Pred-fossil<OBJ> smilodon ','<SUB> distinguished service medal (united states navy)<PRED>-Pred-higher<OBJ> department of commerce gold medal ','<SUB> alan shepard<PRED>-Pred-deathPlace<OBJ> california ','<SUB> california<PRED>-Pred-senators<OBJ> dianne feinstein ', '<SUB> alan shepard<PRED>-Pred-awards<OBJ> distinguished service medal (united states navy)']
    #predicates = ['-Pred-gemstone', '-Pred-fossil', '-Pred-higher', '-Pred-deathPlace', '-Pred-senators', '-Pred-awards']

    #triples = ['<SUB> apollo 12<PRED>-Pred-backup_pilot<OBJ> alfred worden ','<SUB> alan bean<PRED>-Pred-was_a_crew_member_of<OBJ> apollo 12 ','<SUB> apollo 12<PRED>-Pred-operator<OBJ> nasa ','<SUB> alan bean<PRED>-Pred-dateOfRetirement<OBJ> "june 1981" ','<SUB> apollo 12<PRED>-Pred-commander<OBJ> david scott ','<SUB> alan bean<PRED>-Pred-birthName<OBJ> "alan lavern bean"']
    #predicates = ['-Pred-backup_pilot', '-Pred-was_a_crew_member_of', '-Pred-operator', '-Pred-dateOfRetirement', '-Pred-commander', '-Pred-birthName']

    #triples = ['<SUB> buzz aldrin<PRED>-Pred-birthPlace<OBJ> glen ridge, new jersey ', '<SUB> buzz aldrin<PRED>-Pred-was_a_crew_member_of<OBJ> apollo 11 ','<SUB> buzz aldrin<PRED>-Pred-nationality<OBJ> united states ', '<SUB> buzz aldrin<PRED>-Pred-occupation<OBJ> fighter pilot ', '<SUB> apollo 11<PRED>-Pred-backup_pilot<OBJ> william anders ', '<SUB> apollo 11<PRED>-Pred-operator<OBJ> nasa']
    #predicates = ['-Pred-birthPlace', '-Pred-was_a_crew_member_of', '-Pred-nationality', '-Pred-occupation', '-Pred-backup_pilot', '-Pred-operator']

    #triples = ['<SUB> acharya institute of technology<PRED>-Pred-city<OBJ> bangalore ', '<SUB> acharya institute of technology<PRED>-Pred-state<OBJ> karnataka ', '<SUB> acharya institute of technology<PRED>-Pred-country<OBJ> "india" ', '<SUB> acharya institute of technology<PRED>-Pred-numberOfPostgraduateStudents<OBJ> 700 ', '<SUB> acharya institute of technology<PRED>-Pred-campus<OBJ> "in soldevanahalli, acharya dr. sarvapalli radhakrishnan road, hessarghatta main road, bangalore – 560090." ', '<SUB> acharya institute of technology<PRED>-Pred-affiliation<OBJ> visvesvaraya technological university']
    #predicates = ['-Pred-city', '-Pred-state', '-Pred-country', '-Pred-numberOfPostgraduateStudents', '-Pred-campus', '-Pred-affiliation']

    #triples = ['<SUB> india<PRED>-Pred-largestCity<OBJ> mumbai ', '<SUB> awh engineering college<PRED>-Pred-country<OBJ> india ', '<SUB> awh engineering college<PRED>-Pred-established<OBJ> 2001 ', '<SUB> kerala<PRED>-Pred-leaderName<OBJ> kochi ', '<SUB> awh engineering college<PRED>-Pred-state<OBJ> kerala ', '<SUB> india<PRED>-Pred-river<OBJ> ganges']
    #predicates = ['-Pred-largestCity', '-Pred-country', '-Pred-established', '-Pred-leaderName', '-Pred-state', '-Pred-river']

    #triples = ['<SUB> denmark<PRED>-Pred-leaderName<OBJ> lars l<unk>kke rasmussen ', '<SUB> european university association<PRED>-Pred-headquarters<OBJ> brussels ', '<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-country<OBJ> denmark ', '<SUB> denmark<PRED>-Pred-leaderTitle<OBJ> monarchy of denmark ', '<SUB> school of business and social sciences at the aarhus university<PRED>-Pred-affiliation<OBJ> european university association ', '<SUB> denmark<PRED>-Pred-religion<OBJ> church of denmark']
    #predicates = ['-Pred-leaderName', '-Pred-headquarters', '-Pred-country', '-Pred-leaderTitle', '-Pred-affiliation', '-Pred-religion']

    #triples = ['<SUB> united states<PRED>-Pred-demonym<OBJ> americans ', '<SUB> united states<PRED>-Pred-capital<OBJ> washington, d.c. ', '<SUB> albany, oregon<PRED>-Pred-country<OBJ> united states ', '<SUB> united states<PRED>-Pred-ethnicGroup<OBJ> native americans in the united states ', '<SUB> albany, oregon<PRED>-Pred-isPartOf<OBJ> linn county, oregon']
    #predicates = ['-Pred-demonym', '-Pred-capital', '-Pred-country', '-Pred-ethnicGroup', '-Pred-isPartOf']

    #triples = ['<SUB> ampara hospital<PRED>-Pred-country<OBJ> sri lanka ', '<SUB> sri lanka<PRED>-Pred-leaderName<OBJ> ranil wickremesinghe ', '<SUB> ampara hospital<PRED>-Pred-state<OBJ> eastern province, sri lanka ', '<SUB> eastern province, sri lanka<PRED>-Pred-governingBody<OBJ> eastern provincial council ', '<SUB> sri lanka<PRED>-Pred-capital<OBJ> sri jayawardenepura kotte']
    #predicates = ['-Pred-country', '-Pred-leaderName', '-Pred-state', '-Pred-governingBody', '-Pred-capital']

    triples = ['<SUB> auburn, alabama<PRED>-Pred-isPartOf<OBJ> lee county, alabama ', '<SUB> alabama<PRED>-Pred-country<OBJ> united states ']
    predicates = ['-Pred-isPartOf', '-Pred-country']

    res = cluster_obj.run(predicates, triples)

    print ('\n\n\n', res)
    
