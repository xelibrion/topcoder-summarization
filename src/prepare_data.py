#!/usr/bin/env python

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

from nnet.model_input import InputProcessor


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


iproc = None


def initializer():
    global iproc
    iproc = InputProcessor()


def run_preprocessing(input_tuple):
    def _ndarray_to_list(array):
        assert isinstance(array, np.ndarray)
        return [x.tolist() for x in array]

    tgt, src, or_ids = input_tuple
    tgt, src = _ndarray_to_list(tgt), _ndarray_to_list(src)

    result = iproc.preprocess(
        src,
        tgt,
        or_ids,
    )
    return result


def process(total_items, abstract_tokens, article_tokens, oracle_ids, desc):
    items_processed = []

    with tqdm(total=total_items, desc=desc) as tq:
        with Pool(os.cpu_count(), initializer=initializer) as pool:
            args = list(zip(abstract_tokens, article_tokens, oracle_ids))
            for result in pool.imap(run_preprocessing, args):
                tq.update()
                items_processed.append(result)

            pool.close()
            pool.join()
    return items_processed


def main():
    pq_df = pq.ParquetDataset(rel_path('../data/tokenized_p'))
    df = pq_df.read().to_pandas()

    df_marked = pd.read_json(
        rel_path('../data/selected_sentences.jsonl'),
        lines=True,
        orient='records',
    )

    df = pd.merge(df, df_marked, on='article_id')

    df_train, df_rest = train_test_split(df, test_size=.3)
    df_val, df_test = train_test_split(df_rest, test_size=.5)
    for label, df in zip(['Train', 'Val', 'Test'], [df_train, df_val, df_test]):
        result = process(
            df.shape[0],
            df['abstract_tokens'],
            df['article_tokens'],
            df['selected_sentences_ids'],
            label,
        )
        r_df = pd.DataFrame(result)
        r_df.dropna(inplace=True)
        r_df.to_json(
            rel_path(f'../data/ready/{label.lower()}.jsonl'),
            lines=True,
            orient='records',
        )


if __name__ == '__main__':
    main()
