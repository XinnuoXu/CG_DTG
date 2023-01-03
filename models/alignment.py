#coding=utf8

import json

import wordsegment
wordsegment.load()
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import nltk.data
import spacy
import neuralcoref
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, modified_precision
from torchmetrics.text.bert import BERTScore
from bleurt import score as bleurt_score

class InputOutputAlignment():
    def __init__(self, coreference=False, run_rouge=False, run_bleu=False, run_bertscore=False, run_bleurt=False):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore = BERTScore(device='cuda')
        self.bleurt_scorer = bleurt_score.BleurtScorer('/rds/user/hpcxu1/hpc-work/BLEURT-20')

        self.coreference = coreference
        self.run_rouge = run_rouge
        self.run_bleu = run_bleu
        self.run_bertscore = run_bertscore
        self.run_bleurt = run_bleurt

    def _run_overlap(self, pair, values_freq):
        (src, tgt) = pair
        sum_scores = 0.0
        for item in src:

            if item not in values_freq:
                weight = 1.0
            else:
                weight = 1.0/values_freq[item]
            print (item, '|||', tgt, scores, weight)
            sum_scores += weight * scores
        return sum_scores

    def _run_rouge(self, pair, values_freq):
        (src, tgt) = pair
        sum_scores = 0.0
        for item in src:
            scores = self.rouge_scorer.score(tgt, item)
            rouge_1 = scores['rouge1'].fmeasure
            rouge_2 = scores['rouge2'].fmeasure
            rouge_L = scores['rougeL'].fmeasure
            scores = rouge_1+rouge_2+rouge_L 
            if item not in values_freq:
                weight = 1.0
            else:
                weight = 1.0/values_freq[item]
            sum_scores += weight * scores
        return sum_scores

    def _run_bleu(self, pair, values_freq):
        (src, tgt) = pair
        sum_scores = 0.0
        for item in src:
            #scores = sentence_bleu([item], tgt)
            scores = float(modified_precision([tgt], item, n=2))
            if item not in values_freq:
                weight = 1.0
            else:
                weight = 1.0/values_freq[item]
            sum_scores += weight * scores
        return sum_scores

    def _run_bertscore(self, pairs):
        srcs = [' '.join(pair[0]) for pair in pairs]
        tgts = [pair[1] for pair in pairs]
        scores = self.bertscore(srcs, tgts)
        return scores['f1']

    def _run_bleurt(self, pairs):
        srcs = [' '.join(pair[0]) for pair in pairs]
        tgts = [pair[1] for pair in pairs]
        scores = self.bleurt_scorer.score(references=srcs, candidates=tgts)
        return scores

    def _run_batch(self, pairs, values_freq):
        align_scores = [[] for pair in pairs]
        if self.run_rouge:
            for i, pair in enumerate(pairs):
                score = self._run_rouge(pair, values_freq)
                align_scores[i].append(score)
        if self.run_bleu:
            for i, pair in enumerate(pairs):
                score = self._run_bleu(pair, values_freq)
                align_scores[i].append(score)
        if self.run_bertscore:
            scores = self._run_bertscore(pairs)
            for i, score in enumerate(scores):
                align_scores[i].append(score)
        if self.run_bleurt:
            scores = self._run_bleurt(pairs)
            for i, score in enumerate(scores):
                align_scores[i].append(score)
        return align_scores

    def _run_coreference(self, tgts):
        sentence = ' '.join(tgts)
        doc = nlp(sentence)
        resolved_sentence = doc._.coref_resolved
        if (len(resolved_sentence.split(' ')) - len(sentence.split(' '))) / len(sentence.split(' ')) > 0.3:
            resolved_sentence = sentence
        tgts = self.tokenizer.tokenize(resolved_sentence)
        return tgts

    def _process_src(self, src):
        sub = src.split(' <PRED> ')[0].replace('<SUB> ', '')
        obj = src.split(' <OBJ> ')[1]
        pred = src.split('<PRED> -Pred-')[1].split(' <OBJ> ')[0]
        new_pred = ' '.join(wordsegment.segment(pred))
        return [sub, new_pred, obj]

    def input_to_output(self, srcs, tgts):

        srcs = [self._process_src(src) for src in srcs]
        values_freq = {}
        for src in srcs:
            sub = src[0]
            obj = src[2]
            if sub not in values_freq:
                values_freq[sub] = 0
            values_freq[sub] += 1
            if obj not in values_freq:
                values_freq[obj] = 0
            values_freq[obj] += 1

        if self.coreference:
            previous_tgt = tgts
            tgts = self._run_coreference(tgts)
            if len(tgts) != len(previous_tgt):
                tgts = previous_tgt

        pairs = []
        for src in srcs:
            for tgt in tgts:
                pairs.append((src, tgt.lower()))

        align_scores = self._run_batch(pairs, values_freq)
        score_idx = 0
        alignments = [[] for tgt in tgts]
        for i, src in enumerate(srcs):
            max_score = -1; max_idx = -1
            for j, tgt in enumerate(tgts):
                pair_score = sum(align_scores[score_idx])
                if pair_score > max_score:
                    max_score = pair_score
                    max_idx = j
                score_idx += 1
            alignments[max_idx].append(i)
        return alignments

