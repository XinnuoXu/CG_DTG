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


    def preprocess(self, src, tgt, max_src_sent_length, max_tgt_length, alg, prompt_str):

        src_txt = (' '+self.cls_token+' ').join(src)
        if self.args.add_plan_to_src == 'hard_prompt':
            src_txt = src_txt + ' ' + self.cls_token + ' ' + prompt_str
        elif self.args.add_plan_to_src == 'soft_prompt':
            plan_seg_number = len(prompt_str.split('|||'))
            src_txt = src_txt + ' ' + self.cls_token + ' ||| ' + ' '.join(['-Pred-ROOT']*plan_seg_number)

        if self.args.add_plan_to_tgt == 'intersec':
            prompts = [' '.join(item.split(' | ')) for item in prompt_str.split(' ||| ')]
            sents = []; new_format_prompts = []
            for i, sent in enumerate(tgt):
                prompt = '| ' + prompts[i] + ' ||| '
                sent = prompt + ' '.join(sent) + ' ' + self.cls_token
                sents.append(sent)
                new_format_prompts.append(prompt)
            tgt_txt = ' '.join(sents)
            prompt_str = ' '.join(new_format_prompts)
        elif self.args.add_plan_to_tgt == 'prompt':
            tgt_txt = (' '+self.cls_token+' ').join([' '.join(sent) for sent in tgt]) + ' ' + self.cls_token
            tgt_txt = prompt_str + ' ' + self.cls_token + ' ' + tgt_txt
        else:
            # self.args.add_plan_to_tgt is 'none'
            tgt_txt = (' '+self.cls_token+' ').join([' '.join(sent) for sent in tgt]) + ' ' + self.cls_token

        if self.args.tokenizer.startswith('t5-'):
            src_txt = self.cls_token + ' ' + src_txt
            tgt_txt = self.bos_token + ' ' + tgt_txt

        source_tokens = self.tokenizer(src_txt, padding='do_not_pad', truncation=True, max_length=max_src_sent_length)['input_ids']
        target_tokens = self.tokenizer(tgt_txt, padding='do_not_pad', truncation=True, max_length=max_tgt_length)['input_ids']
        prompt_tokens = self.tokenizer(prompt_str, padding='do_not_pad', truncation=True, max_length=max_tgt_length)['input_ids']

        if self.args.for_stepwise:
            target_tokens = target_tokens[:-1]

        #print (src_txt, source_tokens)
        #print (tgt_txt, target_tokens)

        cls_ids = [i for i, t in enumerate(source_tokens) if t == self.cls_token_id]

        new_alg = []
        for sent in alg:
            s = []
            for fact in sent:
                s.extend(fact)
            if len(s) > 0:
                new_alg.append(s)

        return source_tokens, target_tokens, prompt_tokens, cls_ids, src, [' '.join(sent) for sent in tgt], new_alg


def _process(params):

    corpus_type, json_file, args, save_file = params
    logger.info('Processing %s' % json_file)
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    additional_tokens = None
    if args.additional_token_path != '':
        additional_tokens = [line.strip() for line in open(args.additional_token_path)]

    bert = BertData(args, additional_tokens)
    jobs = json.load(open(json_file))

    datasets = []; max_src_len = 0; max_tgt_len = 0
    for d in jobs:
        eid = d['example_id']
        src = d['src'] #[sent1, sent2, sent3...]
        tgt = d['tgt'] #[[seg1, seg2...], [seg1, seg2...]...]
        alg = d['alignments']
        prompt_str = d['prompt_str']

        b_data = bert.preprocess(src, tgt, args.max_src_ntokens, args.max_tgt_ntokens, alg, prompt_str)
        source_tokens, target_tokens, prompt_tokens, cls_ids, src_txt, tgt_txt, alg = b_data

        b_data_dict = {"src": source_tokens, "tgt": target_tokens, "clss": cls_ids,
                       "src_txt": src_txt, "tgt_txt": tgt_txt, 
                       "nsent_src":len(src), "nsent_tgt":len(tgt), 
                       "alignments": alg, "prompt_str":prompt_str, 
                       "prompt_tokenized": prompt_tokens,
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
            new_obj['alignments'] = json_obj['oracles_selection']
            new_obj['predicates'] = json_obj['predicates']
            new_obj['prompt_str'] = json_obj['prompt_str']
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
