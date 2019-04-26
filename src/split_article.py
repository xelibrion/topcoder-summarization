#!/usr/bin/env python
import os
from tqdm import tqdm
import spacy
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer

from multiprocessing import Pool

BERT_MODEL = 'bert-base-uncased'


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def initializer():
    global nlp, tokenizer
    nlp = spacy.load('en_core_web_lg')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


def split_sentence(input_tuple):
    article_id, article = input_tuple
    doc = nlp(article)
    return [(article_id, sent.text, len(tokenizer.tokenize(sent.text)))
            for sent in doc.sents]


df = pd.read_json(
    rel_path('../data/unpacked.jsonl'),
    lines=True,
    orient='records',
)
articles = df['article']

split_sentences = []

with tqdm(total=len(articles)) as tq:
    with Pool(12, initializer=initializer) as pool:

        args = list(enumerate(articles))
        for article_sentences in pool.imap_unordered(split_sentence, args):
            tq.update()
            for idx, t in enumerate(article_sentences):
                art_id, text, num_tokens = t
                split_sentences.append((art_id, idx, text, num_tokens))

        pool.close()
        pool.join()

df_result = pd.DataFrame(split_sentences,
                         columns=[
                             'article_id',
                             'sequence_id',
                             'sentence',
                             'num_bert_tokens',
                         ])
print(df_result.head(3))
df_result.to_json(
    rel_path('../data/sentences.jsonl'),
    lines=True,
    orient='records',
)
