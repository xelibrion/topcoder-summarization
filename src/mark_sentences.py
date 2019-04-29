import os
import argparse
import re
import itertools
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
import spacy


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def perform_selection(input_tuple):
    article_id, abstract_sents, article_sents = input_tuple
    abstract_sents_tokens = [[t.text for t in nlp(s)] for s in abstract_sents]
    article_sents_tokens = [[t.text for t in nlp(s)] for s in article_sents]
    selected_sentences_ids = greedy_selection(
        article_sents_tokens,
        abstract_sents_tokens,
        3,
    )
    return article_id, selected_sentences_ids


def initializer():
    global nlp
    nlp = spacy.load('en_core_web_sm')


def process(total_items, article_id, abstracts, articles):
    items_processed = []

    with tqdm(total=total_items) as tq:
        with Pool(os.cpu_count(), initializer=initializer) as pool:
            args = list(zip(article_id, abstracts, articles))
            for result in pool.imap(perform_selection, args):
                tq.update()
                items_processed.append(result)

            pool.close()
            pool.join()
    return items_processed


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--input_path',
        type=str,
        default='../data/train_sentences.jsonl',
    )
    args = parser.parse_args()

    df = pd.read_json(
        rel_path('../data/train_sentences.jsonl'),
        lines=True,
        orient='records',
    )
    items_processed = process(
        df.shape[0],
        df['article_id'],
        df['abstract_sentences'],
        df['article_sentences'],
    )
    print(items_processed[:10])

    df = pd.DataFrame(items_processed, columns=[
        'article_id',
        'selected_sentences_ids',
    ])
    df.to_json(
        rel_path('../data/selected_sentences.jsonl'),
        lines=True,
        orient='records',
    )


if __name__ == '__main__':
    main()