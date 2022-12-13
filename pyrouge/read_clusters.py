#coding=utf8

import sys
sys.path.append('./models')
from sklearn import metrics
from spectral_clustering import SpectralCluser
import random
import json

def run_metrics(pred, gt):
    pred_dict = {}
    for i, item in enumerate(pred):
        if len(item.strip()) == 0:
            continue
        for predicate in item.split(' '):
            pred_dict[predicate] = i

    gt_dict = {}
    for i, item in enumerate(gt):
        if len(item.strip()) == 0:
            continue
        for predicate in item.split(' '):
            gt_dict[predicate] = i

    if len(gt_dict) != len(pred_dict):
        return None

    pred_label = []
    gt_label = []
    for key in pred_dict:
        pred_label.append(pred_dict[key])
        gt_label.append(gt_dict[key])

    ARS = metrics.adjusted_rand_score(gt_label, pred_label)
    NMI = metrics.normalized_mutual_info_score(gt_label, pred_label) 
    AMI = metrics.adjusted_mutual_info_score(gt_label, pred_label)
    FMS = metrics.fowlkes_mallows_score(gt_label, pred_label)

    return ARS, NMI, AMI, FMS


def generate_random_cluster(predicates, cls_num):
    predicates = ' '.join(predicates).split()
    clusters = [[] for i in range(cls_num)]
    for pred in predicates:
        cls_id = random.randint(0, cls_num-1)
        clusters[cls_id].append(pred)
    return [' '.join(cluster) for cluster in clusters]


def generate_spectual_cluster(predicates, srcs, cls_num):
    predicates = ' '.join(predicates).split()
    clusters = cluster_obj.run(predicates, srcs, gt_ncluster=cls_num)
    return [' '.join(cluster) for cluster in clusters]


if __name__ == '__main__':
    random_clustering = False; spectual_clustering = True

    if spectual_clustering:
        spectual_training_path = '../Plan_while_Generate/D2T_data/webnlg_data/train.jsonl'
        cluster_obj = SpectralCluser(method = 'spectral_clustering',
                                     assign_labels = 'discretize',
                                     eigen_solver = 'arpack',
                                     affinity = 'precomputed',
                                     max_group_size = 10,
                                     min_pair_freq = 15,
                                     use_ratio = False,
                                     filter_with_entities = True,
                                     train_file = spectual_training_path)

    src_filename = '../Plan_while_Generate/D2T_data/webnlg_data/test.jsonl'
    srcs = {}
    for line in open(src_filename):
        json_obj = json.loads(line.strip())
        ref_id = json_obj['example_id']
        s = json_obj['document_segs']
        srcs[ref_id] = s

    filename = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.marginal.kmeans/test.res.18000.cluster'
    #filename = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.marginal.planemb/test.res.20000.cluster'
    examples = {}
    for line in open(filename):
        flist = line.strip('\n').split('\t')
        eid = flist[0]
        score = flist[1]
        pred = flist[2]
        gt = flist[3]

        example_id = eid.split('_')[0]
        reference_id = eid.split('_')[1]
        if example_id not in examples:
            examples[example_id] = {}
        if reference_id not in examples[example_id]:
            examples[example_id][reference_id] = {}
            examples[example_id][reference_id]['scores'] = []
            examples[example_id][reference_id]['predictions'] = []
            examples[example_id][reference_id]['ground_truth'] = []
        examples[example_id][reference_id]['scores'].append(score)
        examples[example_id][reference_id]['predictions'].append(pred)
        examples[example_id][reference_id]['ground_truth'].append(gt)

    ARS = []
    NMI = []
    AMI = []
    FMS = []

    for idx in examples:
        example = examples[idx]
        predictions = []
        ground_truthes = []
        scores = []
        for ref_id in example:
            ground_truthes.append(example[ref_id]['ground_truth'])
            if random_clustering:
                predictions.append(generate_random_cluster(example[ref_id]['predictions'], len(example[ref_id]['ground_truth'])))
            elif spectual_clustering:
                idx_id = f'{idx}_{ref_id}'
                predictions.append(generate_spectual_cluster(example[ref_id]['predictions'], srcs[idx_id], len(example[ref_id]['ground_truth'])))
            else:
                predictions.append(example[ref_id]['predictions'])
            scores.append(example[ref_id]['scores'])
        for pred in predictions:
            best_metrics = None
            for gt in ground_truthes:
                if len(pred) != len(gt):
                    continue
                current_metrics = run_metrics(pred, gt)
                if current_metrics is None:
                    continue
                if best_metrics is None:
                    best_metrics = list(current_metrics)
                else:
                    for j, metric in enumerate(list(current_metrics)):
                        if best_metrics[j] < metric:
                            best_metrics[j] = metric
            if best_metrics is None:
                continue
            ARS.append(best_metrics[0])
            NMI.append(best_metrics[1])
            AMI.append(best_metrics[2])
            FMS.append(best_metrics[3])

    print ('ARS:', sum(ARS)/len(ARS))
    print ('NMI:', sum(NMI)/len(NMI))
    print ('AMI:', sum(AMI)/len(AMI))
    print ('FMS:', sum(FMS)/len(FMS))
