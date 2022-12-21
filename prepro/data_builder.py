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
import numpy as np
from multiprocess import Pool
from transformers import AutoTokenizer

from models.logging import logger


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


    def preprocess_tgt(self, tgt, max_tgt_length):

        tgt_txt = ' '.join(tgt)

        if self.args.tokenizer.startswith('t5-'):
            tgt_txt = self.bos_token + ' ' + tgt_txt

        target_tokens = self.tokenizer(tgt_txt, 
                                       padding='do_not_pad', 
                                       truncation=True, 
                                       max_length=max_tgt_length)['input_ids']

        return target_tokens, tgt_txt



def split_shard(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test', 'validation']

    for corpus_type in datasets:

        input_path = os.path.join(args.raw_path, corpus_type+'.jsonl')

        json_objs = []
        for line in open(input_path):
            json_objs.append(json.loads(line.strip()))

        dataset = []; p_ct = 0
        for d in json_objs:
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    #for line in dataset:
                    #    save.write(line+'\n')
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        if (len(dataset) > 0):
            pt_file = "{:s}/{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                #for line in dataset:
                #    save.write(line+'\n')
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []



def _process_d2t_base(params):

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

    datasets = []
    for d in jobs:
        eid = d['example_id']

        src = d['document_segs']
        source_tokens, src_txt = data_obj.preprocess_src(src, args.max_src_ntokens)

        tgt = d['gold_segs']
        target_tokens, tgt_txt = data_obj.preprocess_tgt(tgt, args.max_tgt_ntokens)

        b_data_dict = {"src": source_tokens, 
                       "tgt": target_tokens,
                       "src_txt": src_txt,
                       "tgt_txt": tgt_txt,
                       "nsent_src":len(src),
                       "nsent_tgt":len(tgt),
                       "eid": eid}

        datasets.append(b_data_dict)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_for_d2t_base(args):
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
        for d in pool.imap(_process_d2t_base, a_lst):
            pass
        pool.close()


def _process_sentence_level(params):

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

    datasets = []
    for d in jobs:
        eid = d['example_id']

        src = d['document_segs']

        if args.remove_noise_datapoints and corpus_type != 'test':
            if sum([len(group) for group in d['oracles_selection']]) < len(src):
                continue
            if min([len(group) for group in d['oracles_selection']]) == 0:
                continue

        if args.remove_single_triple_datapoints and corpus_type != 'test':
            if len(src) < 2:
                continue

        source_tokens = []
        for s in src:
            source_token, _ = data_obj.preprocess_src([s], args.max_src_ntokens)
            source_tokens.append(source_token)
        src_txt = src

        tgt = d['gold_segs']
        target_tokens = []
        for t in tgt:
            target_token, _ = data_obj.preprocess_tgt([t], args.max_tgt_ntokens)
            target_tokens.append(target_token)
        tgt_txt = tgt

        # tokenize predicates
        predicates = d['predicates']
        predicates_ids = data_obj.tokenizer.convert_tokens_to_ids(predicates)
        predicates_txt = ' '.join(predicates)

        pred_to_sentence = []
        for sentence_alg in d['oracles_selection']:
            pred_to_sentence.append([predicates_ids[idx] for idx in sentence_alg])

        b_data_dict = {"src": source_tokens, 
                       "tgt": target_tokens,
                       "pred": predicates_ids,
                       "p2s": pred_to_sentence,
                       "src_txt": src_txt,
                       "tgt_txt": tgt_txt,
                       "pred_txt": predicates_txt,
                       "eid": eid}

        datasets.append(b_data_dict)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_sentence_level(args):
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
        for d in pool.imap(_process_sentence_level, a_lst):
            pass
        pool.close()

