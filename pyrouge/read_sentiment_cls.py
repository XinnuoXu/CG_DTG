#coding=utf8

import numpy as np
import os, sys
import json
from os.path import join as pjoin
from sklearn.metrics import precision_recall_fscore_support

def check_cluster_to_cls_data():
    raw_dir = "/rds/user/hpcxu1/hpc-work/outputs.ama/cluster2cls/"
    verdict_pred = []
    pros_pred = []
    cons_pred = []
    for filename in os.listdir(raw_dir):
        if not filename.startswith('test'):
            continue
        for line in open(f'{raw_dir}/{filename}'):
            line = line.strip()
            json_obj = json.loads(line)
            for example in json_obj:
                verdict_pred.extend(example['verdict_labels'])
                pros_pred.extend(example['pros_labels'])
                cons_pred.extend(example['cons_labels'])
    print (len(verdict_pred), sum(verdict_pred))
    print (len(pros_pred), sum(pros_pred))
    print (len(cons_pred), sum(cons_pred))

def preprocess_scores(pros_scores, cons_scores):
    for i in range(len(pros_scores)):
        if pros_scores[i] - cons_scores[i] >= 0.0:
            cons_scores[i] = -1
        elif cons_scores[i] - pros_scores[i] >= 0.0:
            pros_scores[i] = -1
    return pros_scores, cons_scores

def rank_and_select(clusters, sent_scores, tag, top_k, example_id):
    sent_scores = np.array(sent_scores)
    selected_ids = np.argsort(-sent_scores)
    top_k = min(top_k, len(clusters))
    example_ids = []; srcs = []; scores = []
    preds = [0] * len(sent_scores)
    for selected_idx in selected_ids[:top_k]:
        if sent_scores[selected_idx] < 0:
            continue
        src = clusters[selected_idx]
        idx = f"{example_id}_Cluster{selected_idx}_{tag}"
        example_ids.append(idx)
        srcs.append(src)
        preds[selected_idx] = 1
        scores.append(sent_scores[selected_idx])
    return example_ids, srcs, scores, preds
    

def preprocess_for_later_version(json_f):
    example_map = {}
    for line in open(json_f):
        json_obj = json.loads(line.strip())
        src_cluster = json_obj['clusters'][0]
        verdict_label = json_obj['verdict_labels'][0]
        pros_label = json_obj['pros_labels'][0]
        cons_label = json_obj['cons_labels'][0]
        verdict_score = json_obj['verdict_scores'][0]
        pros_score = json_obj['pros_scores'][0]
        cons_score = json_obj['cons_scores'][0]
        example_id = json_obj['example_id']
        id_list = example_id.split('_')
        example_id = '_'.join(id_list[:2])
        cluster_id = id_list[-1]
        
        if example_id not in example_map:
            example_map[example_id] = {}
            example_map[example_id]['clusters'] = []
            example_map[example_id]['verdict_labels'] = []
            example_map[example_id]['pros_labels'] = []
            example_map[example_id]['cons_labels'] = []
            example_map[example_id]['verdict_scores'] = []
            example_map[example_id]['pros_scores'] = []
            example_map[example_id]['cons_scores'] = []
            example_map[example_id]['example_id'] = example_id

        example_map[example_id]['clusters'].append(src_cluster)
        example_map[example_id]['verdict_labels'].append(verdict_label)
        example_map[example_id]['pros_labels'].append(pros_label)
        example_map[example_id]['cons_labels'].append(cons_label)
        example_map[example_id]['verdict_scores'].append(verdict_score)
        example_map[example_id]['pros_scores'].append(pros_score)
        example_map[example_id]['cons_scores'].append(cons_score)

    return example_map


def preprocess_for_first_version(json_f):
    example_map = {}
    for line in open(json_f):
        json_obj = json.loads(line.strip())
        example_id = json_obj['example_id']
        example_map[example_id] = json_obj
    return example_map


def print_ground_truth(selected_examples):

    def run(labels, clusters, example_id, tag):
        for idx, cluster in enumerate(clusters):
            if labels[idx] == 0:
                continue
            print_tag = f"{example_id}_Cluster{idx}_{tag}"
            print (f"[{print_tag}]")
            print ('\n'.join(cluster))
            print ('\n\n')

    for example_obj in selected_examples:
        example_id = example_obj['example_id']
        clusters = example_obj['clusters']

        labels = example_obj['verdict_labels']
        run(labels, clusters, example_id, 'verdict')
        
        labels = example_obj['pros_labels']
        run(labels, clusters, example_id, 'pros')

        labels = example_obj['cons_labels']
        run(labels, clusters, example_id, 'cons')


def print_topk(selected_examples, ids_set, fpout, pros_or_cons=False):
    verdict_pred = []; verdict_true = []
    pros_pred = []; pros_true = []
    cons_pred = []; cons_true = []
    for example_obj in selected_examples:
        if pros_or_cons:
            example_obj['pros_scores'], example_obj['cons_scores'] = preprocess_scores(example_obj['pros_scores'], example_obj['cons_scores'])
        clusters = example_obj['clusters']
        sent_scores = example_obj['verdict_scores']
        tag = 'verdict'
        top_k = 3
        example_id = example_obj['example_id']
        example_ids, srcs, scores, preds = rank_and_select(clusters, sent_scores, tag, top_k, example_id)
        gtruth = example_obj['verdict_labels']
        verdict_pred.extend(preds)
        verdict_true.extend(gtruth)
        if example_id in ids_set:
            for i, idx in enumerate(example_ids):
                fpout.write(f"[{idx}_outof{len(clusters)}]\t{scores[i]}"+'\n')
                fpout.write('\n'.join(srcs[i]))
                fpout.write('\n\n\n')

        clusters = example_obj['clusters']
        sent_scores = example_obj['pros_scores']
        tag = 'pros'
        top_k = 6
        example_id = example_obj['example_id']
        example_ids, srcs, scores, preds = rank_and_select(clusters, sent_scores, tag, top_k, example_id)
        gtruth = example_obj['pros_labels']
        pros_pred.extend(preds)
        pros_true.extend(gtruth)
        if example_id in ids_set:
            for i, idx in enumerate(example_ids):
                fpout.write(f"[{idx}_outof{len(clusters)}]\t{scores[i]}"+'\n')
                fpout.write('\n'.join(srcs[i]))
                fpout.write('\n\n\n')

        clusters = example_obj['clusters']
        sent_scores = example_obj['cons_scores']
        tag = 'cons'
        top_k = 3
        example_id = example_obj['example_id']
        example_ids, srcs, scores, preds = rank_and_select(clusters, sent_scores, tag, top_k, example_id)
        gtruth = example_obj['cons_labels']
        cons_pred.extend(preds)
        cons_true.extend(gtruth)
        if example_id in ids_set:
            for i, idx in enumerate(example_ids):
                fpout.write(f"[{idx}_outof{len(clusters)}]\t{scores[i]}"+'\n')
                fpout.write('\n'.join(srcs[i]))
                fpout.write('\n\n\n')

    gtruth = np.array(verdict_true)
    preds = np.array(verdict_pred)
    print ('Verdict(P/R/F)', precision_recall_fscore_support(gtruth, preds, average='binary', pos_label=1), len(gtruth), sum(gtruth), len(preds), sum(preds))
    gtruth = np.array(pros_true)
    preds = np.array(pros_pred)
    print ('Pros(P/R/F)', precision_recall_fscore_support(gtruth, preds, average='binary', pos_label=1), len(gtruth), sum(gtruth), len(preds), sum(preds))
    gtruth = np.array(cons_true)
    preds = np.array(cons_pred)
    print ('Cons(P/R/F)', precision_recall_fscore_support(gtruth, preds, average='binary', pos_label=1), len(gtruth), sum(gtruth), len(preds), sum(preds))


if __name__ == '__main__':

    '''
    raw_path = './outputs.ama/logs.selector_sentiment/'
    json_f = pjoin(raw_path, 'test.res.json')
    fpout = open('temp/classification_sentiment.txt', 'w')
    example_map = preprocess_for_first_version(json_f)
    '''

    raw_path = './outputs.ama/logs.selector_cons/'
    json_f = pjoin(raw_path, 'test.res.json')
    fpout = open('temp/classification_v3.txt', 'w')
    example_map = preprocess_for_later_version(json_f)

    '''
    raw_path = './outputs.ama/logs.selector_v3/'
    json_f = pjoin(raw_path, 'test.res.json')
    fpout = open('temp/classification_v3.txt', 'w')
    example_map = preprocess_for_later_version(json_f)

    raw_path = './outputs.ama/logs.selector_v2/'
    json_f = pjoin(raw_path, 'test.res.json')
    fpout = open('temp/classification_v2.txt', 'w')
    example_map = preprocess_for_later_version(json_f)

    raw_path = './outputs.ama/logs.selector/'
    json_f = pjoin(raw_path, 'test.res.json')
    fpout = open('temp/classification_v1.txt', 'w')
    example_map = preprocess_for_first_version(json_f)
    '''


    # Choose examples to evaluate
    ids_set = set()
    for line in open('./pyrouge/example_list.txt'):
        example_id = '_'.join(line.strip().split('_')[:2])
        ids_set.add(example_id)

    selected_examples = []
    for example_id in example_map:
        if example_id not in ids_set:
            continue
        selected_examples.append(example_map[example_id])

    # Process examples
    print_topk(example_map.values(), ids_set, fpout, pros_or_cons=True)
    #print_topk(example_map.values(), ids_set, fpout)
    #print_ground_truth(selected_examples, fpout)

    check_cluster_to_cls_data()
