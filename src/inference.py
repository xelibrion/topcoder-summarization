#!/usr/bin/env python

import argparse
from tqdm import tqdm

from multiprocessing import Pool
import spacy


def initializer():
    global nlp
    nlp = spacy.load('en_core_web_lg')


def split_article(in_tuple):
    idx, article = in_tuple
    doc = nlp(article)
    return idx, [sent.text for sent in doc.sents]


def first3_sentences(input_fname):
    with open(input_fname) as fin:
        articles = [l.strip() for l in fin.readlines()]

        with tqdm(total=len(articles)) as tq:
            with Pool(12, initializer=initializer) as pool:

                args = list(enumerate(articles))
                for idx, sentences in pool.imap(split_article, args):
                    tq.update()
                    if len(sentences) > 6:
                        sentences = sentences[:6]
                    yield ' '.join(['<s> %s .</s>' % s for s in sentences])

                pool.close()
                pool.join()


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'input_path',
        type=str,
    )

    parser.add_argument(
        'output_path',
        type=str,
    )

    args = parser.parse_args()

    with open(args.output_path, 'w') as out_f:
        for abstract in tqdm(first3_sentences(args.input_path)):
            out_f.write(abstract)
            out_f.write('\n')


if __name__ == '__main__':
    main()
