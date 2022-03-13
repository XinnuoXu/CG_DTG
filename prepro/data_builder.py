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


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

    def get_sent_labels(self, src_subtoken_idxs, sent_labels):
        _sent_labels = [0 for t in src_subtoken_idxs if t == self.cls_token_id]
        for l in sent_labels:
            if l >= len(_sent_labels):
                continue
            _sent_labels[l] = 1
        return _sent_labels

    def preprocess(self, src, tgt, sent_labels, max_src_sent_length, max_tgt_length):
        src_txt = (' '+self.cls_token+' ').join(src)
        tgt_txt = (' '+self.sep_token+' ').join([' '.join(sent) for sent in tgt])

        source_tokens = self.tokenizer(src_txt, padding='do_not_pad', truncation=True, max_length=max_src_sent_length)['input_ids']
        target_tokens = self.tokenizer(tgt_txt, padding='do_not_pad', truncation=True, max_length=max_tgt_length)['input_ids']

        gt_selection = self.get_sent_labels(source_tokens, sent_labels)
        cls_ids = [i for i, t in enumerate(source_tokens) if t == self.cls_token_id]

        return source_tokens, target_tokens, gt_selection, cls_ids, src, [' '.join(sent) for sent in tgt]


def _process(params):

    corpus_type, json_file, args, save_file = params
    logger.info('Processing %s' % json_file)
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)
    jobs = json.load(open(json_file))

    datasets = []
    for d in jobs:
        eid = d['example_id']
        sent_labels = d['selections']
        src = d['src'] #[sent1, sent2, sent3...]
        tgt = d['tgt'] #[[seg1, seg2...], [seg1, seg2...]...]

        b_data = bert.preprocess(src, tgt, sent_labels, args.max_src_ntokens, args.max_tgt_ntokens)
        source_tokens, target_tokens, gt_selection, cls_ids, src_txt, tgt_txt = b_data

        b_data_dict = {"src": source_tokens, "tgt": target_tokens,
                       "gt_selection": gt_selection, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "eid": eid}

        datasets.append(b_data_dict)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
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
            selected_segs = set()
            for sent in json_obj['oracles_selection']:
                for seg in sent:
                    selected_segs |= set(seg[:args.oracle_topn])
            new_obj['selections'] = sorted(list(selected_segs))
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
