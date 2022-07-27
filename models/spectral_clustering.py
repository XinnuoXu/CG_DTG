#coding=utf8

import argparse
import json
from sklearn.cluster import SpectralClustering
import numpy as np

FIRST_SENT_LABEL='<FIRST_SENT>'

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
                       pair_frequency_threshold=0.0,
                       max_group_size=3,
                       min_pair_freq=20,
                       use_ratio=False,
                       filter_with_entities=False,
                       train_file='', 
                       valid_file='',
                       test_file=''):

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
            groups = json_obj['prompt_str'].split(' ||| ')
            for i, group in enumerate(groups):
                group = group.split(' ')
                if i == 0:
                    group.append(FIRST_SENT_LABEL)
                edge_weights = self.one_sentence(edge_weights, group)
        return edge_weights

    def train_freq_to_ration(self, edge_weights):
        ratio_edge_weights = {}
        for pred_1 in edge_weights:
            if pred_1 == FIRST_SENT_LABEL:
                pred_1_freq = 0
            else:
                pred_1_freq = sum(edge_weights[pred_1].values())
            ratio_edge_weights[pred_1] = {}
            for pred_2 in edge_weights[pred_1]:
                if pred_2 == FIRST_SENT_LABEL:
                    pred_2_freq = 0
                else:
                    pred_2_freq = sum(edge_weights[pred_2].values())
                ratio_edge_weights[pred_1][pred_2] = ((edge_weights[pred_1][pred_2])*2)/(pred_1_freq+pred_2_freq)
        return ratio_edge_weights


    def process(self, predicates, pred_to_subs_objs, n_clusters):
        #ajacency_matrix = np.ones((len(predicates),len(predicates))) - np.identity(len(predicates))
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
                if self.filter_with_entities and len(pred_to_subs_objs[head_1]&pred_to_subs_objs[head_2]) == 0:
                    ajacency_matrix[i][j] = 0.0
                if ajacency_matrix[i][j] < self.pair_frequency_threshold:
                    ajacency_matrix[i][j] = 0.0

        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        assign_labels=self.assign_labels,
                                        eigen_solver=self.eigen_solver,
                                        affinity=self.affinity).fit(ajacency_matrix)
        return clustering.labels_, ajacency_matrix


    def postprocess(self, candidates, pred_to_subs_objs):

        candidate_map = {}
        for candidate in candidates:
            print ('*******')
            print (candidate)

            # if the candidate contains a very big group or empty group
            bad_group = False
            for group in candidate:
                if len(group) > self.max_group_size:
                    bad_group = True
                    break
            if bad_group:
                continue

            '''
            # if the candidate contains groups whose subs/objs don't overlap
            connected_group = True
            for group in candidate:
                if len(group) == 1:
                    continue
                sub_obj_to_pred = {}
                for pred in group:
                    sub_obj = pred_to_subs_objs[pred]
                    if sub_obj[0] not in sub_obj_to_pred:
                        sub_obj_to_pred[sub_obj[0]] = []
                    if sub_obj[1] not in sub_obj_to_pred:
                        sub_obj_to_pred[sub_obj[1]] = []
                    sub_obj_to_pred[sub_obj[0]].append(pred)
                    sub_obj_to_pred[sub_obj[1]].append(pred)

                independent_pred = set()
                for sub_obj in sub_obj_to_pred:
                    if len(sub_obj_to_pred[sub_obj]) > 1:
                        continue
                    if sub_obj_to_pred[sub_obj][0] in independent_pred:
                        connected_group = False
                        break
                    independent_pred.add(sub_obj_to_pred[sub_obj][0])

            if not connected_group:
                continue
            '''

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
                        print ('Pair frequence:', pred_1, pred_2, freq)
                        if freq < group_min_freq:
                            group_min_freq = freq
                print ('Group lowest frequence:', group, group_min_freq)
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
                if pred in self.model[FIRST_SENT_LABEL]:
                    freq_sum += self.model[FIRST_SENT_LABEL][pred]
                else:
                    freq_sum += 0
            first_sent_freq[i] = freq_sum / float(len(group)) 
        new_best_candidate = []
        for item in sorted(first_sent_freq.items(), key = lambda d:d[1], reverse=True):
            sentence_idx = item[0]
            new_best_candidate.append(best_candidate[sentence_idx])
        return new_best_candidate


    def triples_to_entityes(self, triples, predicates):
        pred_to_subs_objs = {}
        for i, triple in enumerate(triples):
            pred = predicates[i]
            sub_idx = triple.find('<SUB>')
            obj_idx = triple.find('<OBJ>')
            pred_idx = triple.find('<PRED>')
            sub = triple[sub_idx+5:pred_idx].strip()
            obj = triple[obj_idx+5:].strip()
            pred_to_subs_objs[pred] = set([sub, obj])
        return pred_to_subs_objs


    def test(self, predicates, triples):

        candidates = []; ajacency_matrix=None
        if len(predicates) == 1:
            return [predicates]

        pred_to_subs_objs = self.triples_to_entityes(triples, predicates)

        for n_clusters in range(1, len(predicates)+1):
            labels, ajacency_matrix = self.process(predicates, pred_to_subs_objs, n_clusters)
            pred_groups = [[] for i in range(n_clusters)]
            for i, label in enumerate(labels):
                pred_groups[label].append(predicates[i])
            candidates.append(pred_groups)
        print (ajacency_matrix)

        if len(candidates) == 0:
            best_candidate = [predicates]
        else:
            best_candidate = self.postprocess(candidates, pred_to_subs_objs)
        sorted_best_candidate = self.sort_the_groups(best_candidate)

        return sorted_best_candidate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-assign_labels", default='discretize', type=str)
    parser.add_argument("-eigen_solver", default='arpack', type=str)
    parser.add_argument("-affinity", default='precomputed', type=str)
    parser.add_argument("-pair_frequency_threshold", default=0.0, type=float)
    parser.add_argument("-max_group_size", default=3, type=int)
    parser.add_argument("-min_pair_freq", default=20, type=int)
    parser.add_argument("-use_ratio", type=str2bool, default=False)
    parser.add_argument("-filter_with_entities", type=str2bool, default=False)
    parser.add_argument("-train_file", default='../Plan_while_Generate/D2T_data/webnlg_data/train.jsonl', type=str)
    parser.add_argument("-valid_file", default='../Plan_while_Generate/D2T_data/webnlg_data/validation.jsonl', type=str)
    parser.add_argument("-test_file", default='../Plan_while_Generate/D2T_data/webnlg_partial_src/test.jsonl', type=str)
    args = parser.parse_args()

    cluster_obj = SpectralCluser(args.assign_labels,
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

    triples = ['<SUB> 103 colmore row<PRED>-Pred-location<OBJ> birmingham ', '<SUB> 103 colmore row<PRED>-Pred-completionDate<OBJ> 1976 ', '<SUB> 103 colmore row<PRED>-Pred-buildingStartDate<OBJ> "1973" ', '<SUB> 103 colmore row<PRED>-Pred-floorCount<OBJ> 23']
    predicates = ['-Pred-location', '-Pred-completionDate', '-Pred-buildingStartDate', '-Pred-floorCount']

    res = cluster_obj.test(predicates, triples)

    print ('\n\n\n', res)
    
