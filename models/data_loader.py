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

    def __init__(self, data=None, device=None, is_test=False, 
                 pad_id=None, cls_id=None, 
                 pred_special_tok_id=None, 
                 obj_special_tok_id=None,
                 ext_or_abs='abs', 
                 inference_mode='abs',
                 prompt_style='none',
                 shuffle_plan_tok=False):

        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            nsent_src = [x[2] for x in data]
            nsent_tgt = [x[3] for x in data]
            prompt_tokenized = [x[4] for x in data]
            pre_alg = [x[5] for x in data]

            if prompt_style == 'src':
                pre_src = self.prompt_src(pre_src, prompt_tokenized, cls_id, is_test, shuffle_plan_tok)
            elif prompt_style == 'tgt':
                pre_tgt = self.prompt_tgt(pre_tgt, prompt_tokenized, cls_id, is_test)
            elif prompt_style == 'plan_only':
                pre_tgt = prompt_tokenized
            
            src = torch.tensor(self._pad(pre_src, pad_id))
            tgt = torch.tensor(self._pad(pre_tgt, pad_id))
            mask_src = ~(src == pad_id)
            mask_tgt = ~(tgt == pad_id)

            if prompt_tokenized[0] == None:
                prompt_tokenized = None
            else:
                prompt_tokenized = [torch.tensor(item).to(device) for item in prompt_tokenized]

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'nsent_tgt', nsent_tgt)
            setattr(self, 'nsent_src', nsent_src)
            setattr(self, 'prompt_tokenized', prompt_tokenized)
            setattr(self, 'alignments', pre_alg)

            if (ext_or_abs in ['step']) or (inference_mode in ['plan']):
                src_sentence_mask = self.create_sentlevel_mask_src(src, mask_src, cls_id, max(nsent_src))
                tgt_sentence_mask = self.create_sentlevel_mask_tgt(tgt, mask_tgt, cls_id, max(nsent_tgt))
                src_predicate_mask = self.create_predicate_mask_src(src, mask_src, pred_special_tok_id, obj_special_tok_id, max(nsent_src))
                src_predicate_token_idx = self.get_src_predicate_idx(src, pred_special_tok_id)
                src_predicate_token_idx = [item.to(device) for item in src_predicate_token_idx]
                setattr(self, 'mask_tgt_sent', tgt_sentence_mask.to(device))
                setattr(self, 'mask_src_sent', src_sentence_mask.to(device))
                setattr(self, 'src_predicate_mask', src_predicate_mask.to(device))
                setattr(self, 'src_predicate_token_idx', src_predicate_token_idx)


            if (is_test):
                src_str = [x[-4] for x in data]
                setattr(self, 'src_str', src_str)
                if prompt_style == 'plan_only':
                    tgt_str = [[x[-2]] for x in data]
                    setattr(self, 'tgt_str', tgt_str)
                else:
                    tgt_str = [x[-3] for x in data]
                    setattr(self, 'tgt_str', tgt_str)
                prompt_str = [x[-2] for x in data]
                setattr(self, 'prompt_str', prompt_str)
                eid = [x[-1] for x in data]
                setattr(self, 'eid', eid)

    def prompt_src(self, pre_src, prompt_tokenized, cls_id, is_test, shuffle_plan_tok):
        new_srcs = []
        for pair in zip(pre_src, prompt_tokenized):
            src = pair[0]
            prompt = pair[1][:-1]
            eos_id = src[-1]
            if (not is_test) and shuffle_plan_tok:
                random.shuffle(prompt)
            src = src[:-1] + [cls_id] + prompt + [eos_id]
            new_srcs.append(src)
        return new_srcs

    def prompt_tgt(self, pre_tgt, prompt_tokenized, cls_id, is_test):
        new_tgts = []
        for pair in zip(pre_tgt, prompt_tokenized):
            tgt = pair[0]
            prompt = pair[1]
            tgt = prompt[:-1] + tgt
            new_tgts.append(tgt)
        return new_tgts

    def create_predicate_mask_src(self, src, mask_src, pred_special_tok_id, obj_special_tok_id, max_sent_len):
        src_sentence_mask = []
        for i, ex_src in enumerate(src):
            pred_index = (ex_src == pred_special_tok_id).nonzero(as_tuple=True)[0]
            obj_index = (ex_src == obj_special_tok_id).nonzero(as_tuple=True)[0]
            mask = torch.zeros((max_sent_len, src.size(1)))
            for j in range(len(pred_index)):
                sid = pred_index[j]+1
                eid = obj_index[j]
                mask[j][sid:eid] = 1
                mask[j] = mask[j] * mask_src[i]
            src_sentence_mask.append(mask)
        return torch.stack(src_sentence_mask)

    def create_sentlevel_mask_src(self, src, mask_src, cls_id, max_sent_len):
        src_sentence_mask = []
        for i, ex_src in enumerate(src):
            cls_index = (ex_src == cls_id).nonzero(as_tuple=True)[0]
            cls_index = cls_index.tolist() + [src.size(1)]
            mask = torch.zeros((max_sent_len, src.size(1)))
            for j in range(len(cls_index)-1):
                sid = cls_index[j]
                eid = cls_index[j+1]
                mask[j][sid:eid] = 1
                mask[j] = mask[j] * mask_src[i]
            src_sentence_mask.append(mask)
        return torch.stack(src_sentence_mask)

    def create_sentlevel_mask_tgt(self, src, mask_src, cls_id, max_sent_len):
        src_sentence_mask = []
        for i, ex_src in enumerate(src):
            cls_index = (ex_src == cls_id).nonzero(as_tuple=True)[0]
            cls_index = cls_index.tolist()
            cls_index[0] = -1
            mask = torch.zeros((max_sent_len, src.size(1)))
            for j in range(len(cls_index)-1):
                sid = cls_index[j]+1
                eid = cls_index[j+1]+1
                mask[j][sid:eid] = 1
                mask[j] = mask[j] * mask_src[i]
            src_sentence_mask.append(mask)
        return torch.stack(src_sentence_mask)

    def get_src_predicate_idx(self, src, pred_special_tok_id):
        src_sentence_mask = []
        for ex_src in src:
            pred_index = (ex_src == pred_special_tok_id).nonzero(as_tuple=True)[0]
            src_sentence_mask.append(pred_index+1)
        return src_sentence_mask

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
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if self.tokenizer.cls_token_id is None:
            self.cls_token = self.tokenizer.eos_token
        else:
            self.cls_token = self.tokenizer.cls_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pred_special_tok_id = self.tokenizer.convert_tokens_to_ids([self.args.pred_special_tok])[0]
        self.obj_special_tok_id = self.tokenizer.convert_tokens_to_ids([self.args.obj_special_tok])[0]

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
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        nsent_tgt = ex['nsent_tgt']
        nsent_src = ex['nsent_src']
        prompt_str = ex['prompt_str']
        prompt_tokenized = ex['prompt_tokenized']
        alg = ex['alignments']

        src = src[:-1][:self.args.max_pos-1]+[src[-1]]
        tgt = tgt[:-1][:self.args.max_tgt_len]+[tgt[-1]]
        prompt_tokenized = prompt_tokenized[:-1][:self.args.max_prompt_len]+[prompt_tokenized[-1]]

        if(is_test):
            return src, tgt, nsent_src, nsent_tgt, prompt_tokenized, alg, src_txt, tgt_txt, prompt_str, eid
        else:
            return src, tgt, nsent_src, nsent_tgt, prompt_tokenized, alg

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
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
                batch = Batch(minibatch, self.device, self.is_test, 
                              self.pad_token_id, cls_id=self.cls_token_id, 
                              pred_special_tok_id=self.pred_special_tok_id, 
                              obj_special_tok_id=self.obj_special_tok_id,
                              ext_or_abs=self.args.ext_or_abs,
                              inference_mode=self.args.inference_mode,
                              prompt_style=self.args.prompt_style,
                              shuffle_plan_tok=self.args.shuffle_plan_tok)
                yield batch
            return


