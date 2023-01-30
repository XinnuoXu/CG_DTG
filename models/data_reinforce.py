import gc
import glob
import bisect
import random
import json
import torch
from models.logging import logger
from transformers import AutoTokenizer
torch.set_printoptions(edgeitems=1000)

class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)

        data = [d[:width] for d in data]
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _process_tgt(self, pre_tgt, pad_id, device):

        def get_conditioned_str(tgt, width):
            conditions = []
            ctgt = []
            mask_ctgt = torch.full((len(tgt), width), False)
            mask_ctgt_loss = torch.full((len(tgt), width), False)
            for i, t in enumerate(tgt):
                if i == 0:
                    tgt_toks = t
                else:
                    tgt_toks = conditions + t[1:]
                mask_ctgt[i][:len(tgt_toks)] = True
                mask_ctgt_loss[i][len(conditions):len(tgt_toks)] = True
                ctgt.append(tgt_toks)
                conditions = tgt_toks
            ctgt = torch.tensor(self._pad(ctgt, pad_id, width=width))
            pad_mask = ~(ctgt == pad_id)
            mask_ctgt = mask_ctgt & pad_mask
            mask_ctgt_loss = mask_ctgt_loss & pad_mask

            return ctgt, mask_ctgt, mask_ctgt_loss

        # Pure tgt sentences
        widths = []
        for ex in pre_tgt:
            widths.append(max([len(sent) for sent in ex]))
        width = max(widths)

        tgt = []; mask_tgt = []
        for i, t in enumerate(pre_tgt):
            # raw_t is tgt sentences of one example
            t = torch.tensor(self._pad(t, pad_id, width=width)).to(device)
            tgt.append(t)

            m_t = ~(t == pad_id).to(device)
            mask_tgt.append(m_t)

        # tgt with conditions
        widths = []
        for ex in pre_tgt:
            widths.append(sum([len(sent) for sent in ex]))
        width = max(widths)

        ctgt = []; mask_ctgt = []; mask_ctgt_loss = []
        for i, t in enumerate(pre_tgt):
            ct, m_ct, m_ct_l = get_conditioned_str(t, width)
            ct = ct.to(device)
            m_ct = m_ct.to(device)
            m_ct_l = m_ct_l.to(device)
            ctgt.append(ct)
            mask_ctgt.append(m_ct)
            mask_ctgt_loss.append(m_ct_l)

        return tgt, mask_tgt, ctgt, mask_ctgt, mask_ctgt_loss


    def _process_predicates(self, preds, preds_tokens, p2s, pad_id, device,
                            all_predicates=None, nn_cls_add_negative_samples=False):


        def negative_sample_for_pred(all_predicates, pred):
            neg_pairs = []; neg_labels = []
            for pid in pred:
                sampled_id = random.sample(all_predicates.keys(), 1)[0]
                while sampled_id in pred:
                    sampled_id = random.sample(all_predicates.keys(), 1)[0]
                if pid < int(sampled_id):
                    pair = all_predicates[str(pid)] + all_predicates[sampled_id]
                else:
                    pair = all_predicates[sampled_id] + all_predicates[str(pid)]
                neg_pairs.append(pair)
                neg_labels.append(0)

            return neg_pairs, neg_labels

        pred_t = []
        pred_mt = []
        agg_labels = []

        max_pair_len = []
        for example_pred_token in preds_tokens:
            max_len = max([len(item) for item in example_pred_token])
            max_pair_len.append(max_len*2)
        max_pair_len = max(max_pair_len)

        for example_id, pred in enumerate(preds):
            pred_tokens = preds_tokens[example_id]

            agg_info = {}; group_size = {}
            for i, group in enumerate(p2s[example_id]):
                for item in group:
                    agg_info[item] = i
                    group_size[item] = len(group)

            pairs = []; labels = []
            for i in range(len(pred)):
                for j in range(len(pred)):
                    head_i = pred[i]
                    head_j = pred[j]
                    if head_i < head_j:
                        pair = pred_tokens[i] + pred_tokens[j]
                    else:
                        pair = pred_tokens[j] + pred_tokens[i]
                    pairs.append(pair)

                    if (head_i not in agg_info) or (head_j not in agg_info):
                        labels.append(0)
                    else:
                        if i == j:
                            if group_size[head_i] == 1:
                                labels.append(1)
                            else:
                                labels.append(0)
                        else:
                            if agg_info[head_i] == agg_info[head_j]:
                                labels.append(1)
                            else:
                                labels.append(0)

            if nn_cls_add_negative_samples and (all_predicates is not None):
                neg_pairs, neg_labels = negative_sample_for_pred(all_predicates, pred)
                pairs = pairs + neg_pairs
                labels = labels + neg_labels

            p = torch.tensor(self._pad(pairs, pad_id, width=max_pair_len)).to(device)
            m_p = ~(p == pad_id).to(device)
            l = torch.tensor(labels).to(device)
            pred_t.append(p)
            pred_mt.append(m_p)
            agg_labels.append(l)

        return pred_t, pred_mt, agg_labels


    def __init__(self, data=None, device=None, 
                 is_test=False, pad_id=None, 
                 all_predicates=None,
                 nn_cls_add_negative_samples=False):

        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_pred = [x[2] for x in data]
            pre_pred_tokens = [x[3] for x in data]
            pre_p2s = [x[4] for x in data]
            pre_nsent = [len(x) for x in pre_tgt]
          
            setattr(self, 'src', pre_src)

            # process tgt
            if is_test:
                tgt = None
                mask_tgt = None
                ctgt = None
                mask_ctgt = None
                mask_ctgt_loss = None
            else:
                tgt, mask_tgt, ctgt, mask_ctgt, mask_ctgt_loss = self._process_tgt(pre_tgt, pad_id, device)

            setattr(self, 'tgt', tgt)
            setattr(self, 'mask_tgt', mask_tgt)
            setattr(self, 'ctgt', ctgt)
            setattr(self, 'mask_ctgt', mask_ctgt)
            setattr(self, 'mask_ctgt_loss', mask_ctgt_loss)

            # process preds
            res = self._process_predicates(pre_pred, pre_pred_tokens, pre_p2s, pad_id, device, 
                                           all_predicates=all_predicates,
                                           nn_cls_add_negative_samples=nn_cls_add_negative_samples)
            pred_tokens, pred_mask_tokens, agg_labels = res
            setattr(self, 'pred', pre_pred)
            setattr(self, 'pred_tokens', pred_tokens)
            setattr(self, 'pred_mask_tokens', pred_mask_tokens)
            setattr(self, 'aggregation_labels', agg_labels)

            setattr(self, 'p2s', pre_p2s)
            setattr(self, 'nsent', pre_nsent)

            if (is_test):
                src_str = [x[-5] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-4] for x in data]
                setattr(self, 'tgt_str', tgt_str)
                pred_str = [x[-3] for x in data]
                setattr(self, 'pred_str', pred_str)
                pred_group_str = [x[-2] for x in data]
                setattr(self, 'pred_group_str', pred_group_str)
                eid = [x[-1] for x in data]
                setattr(self, 'eid', eid)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "validation", "test", "test_unseen"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.input_path + '/' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.input_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def ext_batch_size_fn(new, count):
    src = new[0]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args, dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False, shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if self.tokenizer.cls_token_id is None:
            self.cls_token = self.tokenizer.eos_token
        else:
            self.cls_token = self.tokenizer.cls_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pred_special_tok_id = self.tokenizer.convert_tokens_to_ids([self.args.pred_special_tok])[0]
        self.obj_special_tok_id = self.tokenizer.convert_tokens_to_ids([self.args.obj_special_tok])[0]

        self.all_predicates = None
        if args.nn_cls_add_negative_samples and args.seen_predicate_tokenized_paths != '':
            with open(args.seen_predicate_tokenized_paths) as fpout:
                self.all_predicates = json.loads(fpout.readline().strip())

        self._iterations_this_epoch = 0
        self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        eid = ex['eid']
        src = ex['src']
        tgt = ex['tgt']
        pred = ex['pred']
        pred_tokens = ex['pred_tokens']
        p2s = ex['p2s']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        pred_txt = ex['pred_txt']
        pred_group_txt = ex['pred_group_txt']

        if(is_test):
            return src, tgt, pred, pred_tokens, p2s, src_txt, tgt_txt, pred_txt, pred_group_txt, eid
        else:
            return src, tgt, pred, pred_tokens, p2s

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            p_batch = sorted(buffer, key=lambda x: len(x[1]))
            p_batch = self.batch(p_batch, self.batch_size)
            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test, self.pad_token_id, 
                              all_predicates=self.all_predicates,
                              nn_cls_add_negative_samples=self.args.nn_cls_add_negative_samples)
                yield batch
            return


