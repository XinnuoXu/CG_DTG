import gc
import glob
import bisect
import random
import torch
from models.logging import logger
from transformers import AutoTokenizer
torch.set_printoptions(edgeitems=1000)

class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False, pad_id=None, cls_id=None):

        if data is not None:
            self.batch_size = len(data)
            pre_sentences = []
            for x in data:
                pre_sentences.extend(x[0])
            pre_cluster_sizes = [x[1] for x in data]
            pre_verdict_labels = [x[2] for x in data]
            pre_pros_labels = [x[3] for x in data]
            pre_cons_labels = [x[4] for x in data]
            pre_eid = [x[5] for x in data]
          
            src = torch.tensor(self._pad(pre_sentences, pad_id))
            mask_src = ~(src == pad_id)

            verdict_labels = torch.tensor(self._pad(pre_verdict_labels, 0))
            pros_labels = torch.tensor(self._pad(pre_pros_labels, 0))
            cons_labels = torch.tensor(self._pad(pre_cons_labels, 0))

            setattr(self, 'src', src.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'verdict_labels', verdict_labels.to(device))
            setattr(self, 'pros_labels', pros_labels.to(device))
            setattr(self, 'cons_labels', cons_labels.to(device))
            setattr(self, 'cluster_sizes', pre_cluster_sizes)
            setattr(self, 'eid', pre_eid)

            if is_test:
                pre_clusters = [x[-1] for x in data]
                setattr(self, 'clusters', pre_clusters)

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
    assert corpus_type in ["train", "validation", "test"]

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
    if (len(new) == 4):
        pass
    src, labels = new[0], new[3]
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
        self._iterations_this_epoch = 0
        self.batch_size_fn = ext_batch_size_fn

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if self.tokenizer.cls_token_id is None:
            self.cls_token = self.tokenizer.eos_token
        else:
            self.cls_token = self.tokenizer.cls_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def sample_clusters(self, sentences, verdict_labels, pros_labels, cons_labels, cluster_sizes):
        labels = [(verdict_labels[i] | pros_labels[i]) | cons_labels[i] for i in range(len(verdict_labels))]
        current_length = sum([cluster_sizes[i] for i in range(len(labels)) if labels[i] == 1])
        select_ids = [i for i in range(len(labels))]
        random.shuffle(select_ids)
        for select_id in select_ids:
            if current_length > self.args.max_src_nsent:
                break
            if current_length + cluster_sizes[select_id] > self.args.max_src_nsent:
                continue
            labels[select_id] = 1
            current_length += cluster_sizes[select_id]
        new_sentences = []
        new_verdict_labels = []
        new_pros_labels = []
        new_cons_labels = []
        new_cluster_sizes = []
        for i, lab in enumerate(labels):
            if lab == 0:
                continue
            sid = sum(cluster_sizes[:i])
            eid = sid+cluster_sizes[i]
            new_sentences.extend(sentences[sid:eid])
            new_verdict_labels.append(verdict_labels[i])
            new_pros_labels.append(pros_labels[i])
            new_cons_labels.append(cons_labels[i])
            new_cluster_sizes.append(cluster_sizes[i])
        return new_sentences, new_verdict_labels, new_pros_labels, new_cons_labels, new_cluster_sizes

    def preprocess(self, ex):
        eid = ex['eid']
        sentences = ex['sentences']
        verdict_labels = ex['verdict_labels']
        pros_labels = ex['pros_labels']
        cons_labels = ex['cons_labels']
        cluster_sizes = ex['cluster_sizes']
        clusters = ex['clusters']
        if sum(cluster_sizes) > self.args.max_src_nsent and (not self.is_test):
            res = self.sample_clusters(sentences, verdict_labels, pros_labels, cons_labels, cluster_sizes)
            sentences, verdict_labels, pros_labels, cons_labels, cluster_sizes = res
        if (not self.is_test):
            return  sentences, cluster_sizes, verdict_labels, pros_labels, cons_labels, eid
        else:
            return  sentences, cluster_sizes, verdict_labels, pros_labels, cons_labels, eid, clusters

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            ex = self.preprocess(ex)
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
                batch = Batch(minibatch, self.device, self.is_test, self.pad_token_id, cls_id=self.cls_token_id)
                yield batch
            return


