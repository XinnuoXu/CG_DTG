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
        if key not in gt_dict:
            gt_label.append(len(gt_dict))
        else:
            gt_label.append(gt_dict[key])

    ARS = metrics.adjusted_rand_score(gt_label, pred_label)
    NMI = metrics.normalized_mutual_info_score(gt_label, pred_label) 
    AMI = metrics.adjusted_mutual_info_score(gt_label, pred_label)
    FMS = metrics.fowlkes_mallows_score(gt_label, pred_label)

    return ARS, NMI, AMI, FMS


def load_gold_alignment(gold_path):
    examples = {}
    for line in open(gold_path):
        json_obj = json.loads(line.strip())
        alignments = json_obj['oracles_selection']
        preds = json_obj['predicates']

        eid = json_obj['example_id']
        example_id = eid.split('_')[0]
        reference_id = eid.split('_')[1]

        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][reference_id] = []

        for group in alignments:
            pred_group = ' '.join([preds[pid] for pid in group])
            examples[example_id][reference_id].append(pred_group)
    return examples


def generate_random_cluster(gold_path):
    examples = {}
    for line in open(gold_path):
        json_obj = json.loads(line.strip())
        alignments = json_obj['oracles_selection']
        preds = json_obj['predicates']

        eid = json_obj['example_id']
        example_id = eid.split('_')[0]
        reference_id = eid.split('_')[1]

        cls_num = len(alignments)
        while 1:
            clusters = [[] for i in range(cls_num)]
            for pred in preds:
                cls_id = random.randint(0, cls_num-1)
                clusters[cls_id].append(pred)
            clusters = [' '.join(cluster) for cluster in clusters]
            if min([len(group) for group in clusters]) > 0:
                break

        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][reference_id] = clusters

    return examples


def generate_spectual_cluster(base_path):
    examples = {}
    spectual_training_path = base_path + '/train.jsonl'
    cluster_obj = SpectralCluser(method = 'spectral_clustering',
                                 assign_labels = 'discretize',
                                 eigen_solver = 'arpack',
                                 affinity = 'precomputed',
                                 max_group_size = 10,
                                 min_pair_freq = 15,
                                 use_ratio = False,
                                 filter_with_entities = True,
                                 train_file = spectual_training_path)

    src_filename = base_path + '/test.jsonl'
    for line in open(src_filename):
        json_obj = json.loads(line.strip())

        eid = json_obj['example_id']
        example_id = eid.split('_')[0]
        reference_id = eid.split('_')[1]

        preds = json_obj['predicates']
        srcs = json_obj['document_segs']

        alignments = json_obj['oracles_selection']
        clusters = cluster_obj.run(preds, srcs, gt_ncluster=len(alignments))
        clusters = [' '.join(cluster) for cluster in clusters]

        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][reference_id] = clusters

    return examples


def get_predicted_cluster(pred_path):
    examples = {}
    for line in open(pred_path):
        flist = line.strip('\n').split('\t')
        eid = flist[0]
        score = flist[1]
        pred = flist[2]

        example_id = eid.split('_')[0]
        reference_id = eid.split('_')[1]
        if example_id not in examples:
            examples[example_id] = {}
        if reference_id not in examples[example_id]:
            examples[example_id][reference_id] = []
        examples[example_id][reference_id] = pred.split(' ||| ')
    return examples

if __name__ == '__main__':
    random_clustering = False
    spectual_clustering = False
    manual_alignment = False

    gold_path = '../Plan_while_Generate/D2T_data/webnlg_data/test.jsonl'
    ground_truth_grouping = load_gold_alignment(gold_path)

    if spectual_clustering:
        #base_path = '../Plan_while_Generate/D2T_data/webnlg_data/'
        base_path = '../Plan_while_Generate/D2T_data/webnlg_data.manual_align/'
        examples = generate_spectual_cluster(base_path)

    elif random_clustering:
        examples = generate_random_cluster(gold_path)

    elif manual_alignment:
        manual_path = '../Plan_while_Generate/D2T_data/webnlg_data.manual_align/test.jsonl'
        examples = load_gold_alignment(manual_path)

    else:
        #pred_path = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.base/test.res.2000.cluster'
        #pred_path = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.evenly_mix/test.res.45000.cluster'
        #pred_path = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.joint/test.res.45000.cluster'
        pred_path = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.from_scratch/test.res.5000.cluster'
        #pred_path = '/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.gold_random/test.res.35000.cluster'
        examples = get_predicted_cluster(pred_path)


    ARS = []
    NMI = []
    AMI = []
    FMS = []

    for idx in examples:
        example = examples[idx]
        predictions = []
        ground_truthes = []
        for ref_id in example:
            ground_truth = ground_truth_grouping[idx][ref_id]
            if len(ground_truth) == 1:
                continue
            ground_truthes.append(ground_truth)
            predictions.append(example[ref_id])

        for pred in predictions:
            best_metrics = None
            for gt in ground_truthes:
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
