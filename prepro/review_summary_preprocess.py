#coding=utf8

def ugly_sentence_segmentation(sentence):
    phrases = []
    segs = sentence.split(',')
    for phrase in segs:
        smaller_granularity = phrase.split(' and ')
        for item in smaller_granularity:
            item = item.strip()
            phrases.append(item)
    new_phrases = []
    for i, phrase in enumerate(phrases):
        if i > 0 and len(phrase.split()) < 3:
            new_phrases[-1] = new_phrases[-1] + ' ' + phrase
        else:
            new_phrases.append(phrase)
    if len(new_phrases[0].split()) < 3 and len(new_phrases) > 1:
        first_phrase = new_phrases[0]
        new_phrases = new_phrases[1:]
        new_phrases[0] = first_phrase + ' ' + new_phrases[0]
    return new_phrases


def preprocess_reviews(sentences, high_freq_reviews=None):
    new_sentences = []
    for sentence in sentences:
        search_key = sentence.strip().lower()
        flist = search_key.split()
        search_key = ' '.join([tok for tok in flist if len(tok) > 2])
        if high_freq_reviews != None and search_key in high_freq_reviews:
            continue
        sentence = sentence.lower()
        phrases = ugly_sentence_segmentation(sentence)
        phrases = [item for item in phrases if len(item.strip().split()) > 2]
        new_sentences.extend(phrases)
    return new_sentences


def preprocess_summaries(sentences, do_sentence_segmentation=True):
    new_sentences = []; new_prefix = []
    for sentence in sentences:
        sentence = sentence.lower()
        prefix = ' '.join(sentence.split()[:5])
        sentence = ' '.join(sentence.split()[5:])
        if do_sentence_segmentation:
            phrases = ugly_sentence_segmentation(sentence)
            phrases = [item for item in phrases]
            prefixes = [prefix for item in phrases]
            new_sentences.extend(phrases)
            new_prefix.extend(prefixes)
        else:
            new_sentences.append(sentence)
            new_prefix.append(prefix)
    return new_sentences, new_prefix


