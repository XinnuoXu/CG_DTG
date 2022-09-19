import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import copy
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool
from transformers import AutoTokenizer

from models.logging import logger
from models.spectral_clustering import SpectralCluser

from prepro.dbscan_clustering import DBSCANCluser

class DataCreator():
    def __init__(self, args, additional_tokens=None):

        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        if args.tokenizer.startswith('t5-'):
            special_tokens_dict = {"cls_token": "<s>", "bos_token": "<s>"}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        if additional_tokens is not None:
            print ('The vocab size before adding new tokens: %d' % (len(self.tokenizer)))
            self.tokenizer.add_tokens(additional_tokens)

        self.tokenizer.save_pretrained(args.saved_tokenizer_path)
        print ('The vocab size after adding new tokens: %d' % (len(self.tokenizer)))

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.cls_token = self.tokenizer.cls_token
        self.bos_token = self.tokenizer.bos_token


    def preprocess_src(self, src, max_src_length):

        src_txt = (' '+self.cls_token+' ').join(src)
        # Tokenization using tokenizer
        if self.args.tokenizer.startswith('t5-'):
            src_txt = self.cls_token + ' ' + src_txt
        source_tokens = self.tokenizer(src_txt, padding='do_not_pad', truncation=True, max_length=max_src_length)['input_ids']

        return source_tokens, src


    def preprocess_tgt(self, tgt, tgt_prefix, max_tgt_length):

        #tgt_txt = (' '+self.cls_token+' ').join([' '.join(sent) for sent in tgt]) + ' ' + self.cls_token
        #tgt_prefix_txt = (' '+self.cls_token+' ').join(tgt_prefix) + ' ' + self.cls_token
        tgt_txt = (' '+self.cls_token+' ').join([' '.join(sent) for sent in tgt])
        tgt_prefix_txt = (' '+self.cls_token+' ').join(tgt_prefix)

        # Tokenization using tokenizer
        if self.args.tokenizer.startswith('t5-'):
            tgt_txt = self.bos_token + ' ' + tgt_txt
            tgt_prefix_txt = self.bos_token + ' ' + tgt_prefix_txt

        target_tokens = self.tokenizer(tgt_txt, padding='do_not_pad', truncation=True, max_length=max_tgt_length)['input_ids']
        tgt_prefix_tokens = self.tokenizer(tgt_prefix_txt, padding='do_not_pad', truncation=True, max_length=max_tgt_length)['input_ids']

        if self.args.no_bos_for_tgt:
            target_tokens = target_tokens[1:]
            tgt[0][0] = ' '.join(tgt[0][0].split(' ')[1:])

        if len(tgt_prefix) > 0:
            tgt_prefix_tokens = tgt_prefix_tokens[:-1]
            target_tokens = target_tokens[1:]
        else:
            tgt_prefix_tokens = []

        print (tgt_prefix_tokens, target_tokens)

        return target_tokens, tgt_prefix_tokens, [' '.join(sent) for sent in tgt]


def _process(params):

    corpus_type, json_file, args, save_file = params
    logger.info('Processing %s' % json_file)
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    additional_tokens = None
    if args.additional_token_path != '':
        additional_tokens = [line.strip() for line in open(args.additional_token_path)]

    data_obj = DataCreator(args, additional_tokens)
    jobs = json.load(open(json_file))

    datasets = []; max_src_len = 0; max_tgt_len = 0
    for d in jobs:
        eid = d['example_id']
        src = d['src'] #[sent1, sent2, sent3...]
        tgt = d['tgt'] #[[seg1, seg2...], [seg1, seg2...]...]
        tgt_prefix = []
        if 'tgt_prefix' in d:
            tgt_prefix = d['tgt_prefix']

        source_tokens, src_txt = data_obj.preprocess_src(src, args.max_src_ntokens)
        target_tokens, tgt_prefix_tokens, tgt_txt = data_obj.preprocess_tgt(tgt, tgt_prefix, args.max_tgt_ntokens)

        b_data_dict = {"src": source_tokens, "tgt": target_tokens,
                       'tgt_prefix': tgt_prefix_tokens,
                       "src_txt": src_txt, "tgt_txt": tgt_txt, 
                       "nsent_src":len(src), "nsent_tgt":len(tgt), 
                       "eid": eid}

        datasets.append(b_data_dict)
        max_src_len = max(max_src_len, len(source_tokens))
        max_tgt_len = max(max_tgt_len, len(target_tokens))

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    logger.info('Max src length %d' % max_src_len)
    logger.info('Max tgt length %d' % max_tgt_len)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_for_training(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['validation', 'train', 'test']

    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_process, a_lst):
            pass

        pool.close()
        pool.join()


def split_shard(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'validation']

    for corpus_type in datasets:

        input_path = os.path.join(args.raw_path, corpus_type+'.jsonl')

        json_objs = []
        for line in open(input_path):
            json_obj = json.loads(line.strip())
            new_obj = {}
            new_obj['src'] = json_obj['document_segs']
            new_obj['tgt'] = json_obj['gold_segs']
            new_obj['example_id'] = json_obj['example_id']
            if 'tgt_prefix' in json_obj:
                new_obj['tgt_prefix'] = json_obj['tgt_prefix']
            #print (new_obj['src'], json_obj['tgt_prefix'], new_obj['tgt'], '\n')
            json_objs.append(new_obj)

        dataset = []; p_ct = 0
        for d in json_objs:
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        if (len(dataset) > 0):
            pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def split_shard_spectral_cluster(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'validation']

    cluster_obj = SpectralCluser(method=args.spectral_method,
                                 assign_labels = args.spectral_assign_labels,
                                 eigen_solver = args.spectral_eigen_solver,
                                 affinity = args.spectral_affinity,
                                 max_group_size = args.spectral_max_group_size,
                                 min_pair_freq = args.spectral_min_pair_freq,
                                 use_ratio = args.spectral_use_ratio,
                                 filter_with_entities = args.spectral_filter_with_entities,
                                 train_file = args.spectral_train_file)

    for corpus_type in datasets:
        input_path = os.path.join(args.raw_path, corpus_type+'.jsonl')
        json_objs = []
        for line in open(input_path):
            json_obj = json.loads(line.strip())
            
            srcs = json_obj['document_segs']
            predicates = json_obj['predicates']

            pred_to_triple = {}; pred_to_position = {}
            for i, pred in enumerate(predicates):
                pred_to_triple[pred] = srcs[i]
                pred_to_position[pred] = i

            pred_aggragation = cluster_obj.run(predicates, srcs, prompt_str=json_obj['prompt_str'])

            for i, group in enumerate(pred_aggragation):
                pred_positions = sorted([pred_to_position[pred] for pred in group])
                partial_src = [srcs[pos] for pos in pred_positions]
                #partial_src = [pred_to_triple[pred] for pred in group]

                new_obj = {}
                new_obj['src'] = copy.deepcopy(partial_src)
                new_obj['tgt'] = copy.deepcopy(json_obj['gold_segs'])
                new_obj['example_id'] = json_obj['example_id'] + '_' + str(i)
                new_obj['predicates'] = group

                if i == 0:
                    if len(pred_aggragation) == 1:
                        #new_obj['src'].append('<FULL_SENT>')
                        new_obj['tgt'][0][0] = '<FULL_SENT> ' + new_obj['tgt'][0][0]
                    else:
                        #new_obj['src'].append('<FIRST_SENT>')
                        new_obj['tgt'][0][0] = '<FIRST_SENT> ' + new_obj['tgt'][0][0]
                else:
                    #new_obj['src'].append('<NOT_FIRST_SENT>')
                    new_obj['tgt'][0][0] = '<NOT_FIRST_SENT> ' + new_obj['tgt'][0][0]

                json_objs.append(new_obj)

        dataset = []; p_ct = 0
        for d in json_objs:
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        if (len(dataset) > 0):
            pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _process_prefix_tgt_test(params):

    corpus_type, json_file, args, save_file = params
    logger.info('Processing %s' % json_file)
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    additional_tokens = None
    if args.additional_token_path != '':
        additional_tokens = [line.strip() for line in open(args.additional_token_path)]

    data_obj = DataCreator(args, additional_tokens)
    jobs = json.load(open(json_file))

    datasets = []; max_src_len = 0; max_tgt_len = 0
    for d in jobs:
        eid = d['example_id']
        src = d['src'] #[[sent1, sent2], [sent3...]...]
        tgt = d['tgt'] #[[seg1, seg2...], [seg1, seg2...]...]
        tgt_prefix = []
        if 'tgt_prefix' in d:
            tgt_prefix = d['tgt_prefix']

        source_tokens = []; src_txt = []
        for group in src:
            s_tokens, s_txt = data_obj.preprocess_src(group, args.max_src_ntokens)
            source_tokens.append(s_tokens)
            src_txt.append(s_txt)
        target_tokens, tgt_prefix_tokens, tgt_txt = data_obj.preprocess_tgt(tgt, tgt_prefix, args.max_tgt_ntokens)

        b_data_dict = {"src": source_tokens, "tgt": target_tokens,
                       'tgt_prefix': tgt_prefix_tokens,
                       "src_txt": src_txt, "tgt_txt": tgt_txt, 
                       "nsent_src":len(src), "nsent_tgt":len(tgt), 
                       "eid": eid}

        datasets.append(b_data_dict)
        max_src_len = max(max_src_len, len(source_tokens))
        max_tgt_len = max(max_tgt_len, len(target_tokens))

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    logger.info('Max src length %d' % max_src_len)
    logger.info('Max tgt length %d' % max_tgt_len)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_for_prefix_tgt_test(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['validation', 'train', 'test']

    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_process_prefix_tgt_test, a_lst):
            pass

        pool.close()
        pool.join()


def split_shard_prefix_tgt(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'validation']

    cluster_obj = SpectralCluser(method=args.spectral_method,
                                 assign_labels = args.spectral_assign_labels,
                                 eigen_solver = args.spectral_eigen_solver,
                                 affinity = args.spectral_affinity,
                                 max_group_size = args.spectral_max_group_size,
                                 min_pair_freq = args.spectral_min_pair_freq,
                                 use_ratio = args.spectral_use_ratio,
                                 filter_with_entities = args.spectral_filter_with_entities,
                                 train_file = args.spectral_train_file)

    for corpus_type in datasets:
        input_path = os.path.join(args.raw_path, corpus_type+'.jsonl')
        json_objs = []
        for line in open(input_path):
            json_obj = json.loads(line.strip())
            
            srcs = json_obj['document_segs']
            predicates = json_obj['predicates']

            pred_to_position = {}
            for i, pred in enumerate(predicates):
                pred_to_position[pred] = i

            pred_aggragation = cluster_obj.run(predicates, srcs, prompt_str=json_obj['prompt_str'])

            src_groups = []
            for i, group in enumerate(pred_aggragation):
                pred_positions = sorted([pred_to_position[pred] for pred in group])
                partial_src = [srcs[pos] for pos in pred_positions]
                src_groups.append(partial_src)

            new_obj = {}
            new_obj['src'] = src_groups
            new_obj['tgt'] = json_obj['gold_segs']
            new_obj['example_id'] = json_obj['example_id'] + '_' + str(i)
            new_obj['predicates'] = pred_aggragation

            json_objs.append(new_obj)

        dataset = []; p_ct = 0
        for d in json_objs:
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        if (len(dataset) > 0):
            pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def simple_split_shard(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'validation']

    for corpus_type in datasets:

        input_path = os.path.join(args.raw_path, corpus_type+'.jsonl')

        json_objs = []
        for line in open(input_path):
            json_objs.append(line.strip())

        dataset = []; p_ct = 0
        for d in json_objs:
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    for line in dataset:
                        save.write(line+'\n')
                    p_ct += 1
                    dataset = []

        if (len(dataset) > 0):
            pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                for line in dataset:
                    save.write(line+'\n')
                p_ct += 1
                dataset = []


def format_hdbscan(args):

    input_path = args.raw_path
    output_path = args.save_path
    high_freq_reviews = args.additional_token_path

    dbscan_obj = DBSCANCluser(db_metric='euclidean',
                              sentence_embedding_model='all-MiniLM-L12-v2',
                              db_eps=0.4,
                              db_cluster_size=5,
                              db_input_dimention=56,
                              db_noise_reprocess_similar_topk=3,
                              db_noise_reprocess_threshold=0.6,
                              db_targets_similar_topk=0.2,
                              db_targets_threshold=0.8,
                              high_freq_reviews=high_freq_reviews)

    filename_in = input_path
    filename_out = output_path
    dbscan_obj.run(filename_in, filename_out)


def format_hdbscan_cluster_to_s2s(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'validation']

    for corpus_type in datasets:
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            fpout = open(pjoin(args.save_path, real_name), 'w')
            output_jsons = []
            for line in open(json_f):
                json_obj = json.loads(line.strip())
                tgt_clusters = json_obj['tgt_clusters']
                src_clusters = json_obj['src_clusters']
                example_id = json_obj['example_id']
                for cluster_id in tgt_clusters:
                    if cluster_id == '-1':
                        continue
                    merge_same_prefix = {}
                    for tgt_sent in tgt_clusters[cluster_id]:
                        tgt_prefix = ' '.join(tgt_sent.split()[:5])
                        tgt_text = ' '.join(tgt_sent.split()[5:])
                        if tgt_prefix not in merge_same_prefix:
                            merge_same_prefix[tgt_prefix] = []
                        merge_same_prefix[tgt_prefix].append([tgt_text])
                    for prefix in merge_same_prefix:
                        src = src_clusters[cluster_id]
                        tgt = merge_same_prefix[prefix]
                        tgt_prefix = [prefix]
                        eid = example_id + '_Cluster' + cluster_id + '_' + prefix.split()[0]
                        new_json = {}
                        new_json['src'] = src
                        new_json['tgt'] = tgt
                        new_json['tgt_prefix'] = tgt_prefix
                        new_json['example_id'] = eid
                        output_jsons.append(new_json)
            fpout.write(json.dumps(output_jsons))
            fpout.close()


