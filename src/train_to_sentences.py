#!/usr/bin/env python

import os
import argparse
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
import spacy


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def initializer():
    global nlp
    nlp = spacy.load('en_core_web_lg')


def row_to_sentences(in_tuple):
    idx, abstract, article = in_tuple
    abstract_sents = abstract.split('</s>')
    abstract_sents = [x.replace('<s>', '').strip() for x in abstract_sents if x != '']
    doc = nlp(article)

    return idx, abstract_sents, [sent.text for sent in doc.sents]


def process(total_items, abstracts, articles):
    items_processed = []

    with tqdm(total=total_items) as tq:
        with Pool(os.cpu_count(), initializer=initializer) as pool:
            args = list(zip(range(total_items), abstracts, articles))
            for result in pool.imap(row_to_sentences, args):
                tq.update()
                items_processed.append(result)

            pool.close()
            pool.join()
    return items_processed


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--input_path',
        type=str,
        default='../data/unpacked.jsonl',
    )
    args = parser.parse_args()

    df = pd.read_json(
        rel_path(args.input_path),
        lines=True,
        orient='records',
    )
    items_processed = process(df.shape[0], df['abstract'], df['article'])
    df = pd.DataFrame(items_processed,
                      columns=[
                          'article_id',
                          'abstract_sentences',
                          'article_sentences',
                      ])
    df.sort_values('article_id', inplace=True)
    df.to_json(
        rel_path('../data/train_sentences.jsonl'),
        lines=True,
        orient='records',
    )


if __name__ == '__main__':
    main()
