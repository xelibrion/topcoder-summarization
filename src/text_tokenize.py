#!/usr/bin/env python

import os
import argparse
import subprocess
import json
from contextlib import contextmanager
from shutil import rmtree

import numpy as np
import pandas as pd
from tqdm import tqdm

JAVA_PARAMS = [
    'java',
    '-Xmx10G',
    'edu.stanford.nlp.pipeline.StanfordCoreNLP',
    '-annotators',
    'tokenize,ssplit',
    '-outputFormat',
    'json',
    '-file',
]


@contextmanager
def tempdir(path):
    rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    yield path
    rmtree(path)


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'stanford_nlp_dir',
        type=str,
        help="",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    classpath = os.path.join(os.path.expanduser(args.stanford_nlp_dir),
                             'stanford-corenlp-3.9.2.jar')
    os.environ['CLASSPATH'] = classpath

    df = pd.read_json(
        rel_path('../data/unpacked.jsonl'),
        lines=True,
        orient='records',
    )

    for col in ['abstract', 'article']:
        with tempdir(rel_path(f'../data/tmp_{col}')) as working_dir:
            with open(rel_path(f'./{col}_tokens.jsonl'), 'w') as out_f:

                process(working_dir, df[col], 5000, args.verbose)
                for idx, tokens in read_results(working_dir):
                    result = json.dumps({
                        'index': idx,
                        'tokens': tokens,
                    })

                    out_f.write(result)
                    out_f.write('\n')


def process(working_dir, text_column, max_chunk_size=25000, verbose=False):
    num_items = text_column.shape[0]
    num_chunks = max(1, int(num_items / max_chunk_size))
    print(f'Creating {num_chunks} chunks')

    for idx, chunk in enumerate(
            tqdm(
                np.array_split(text_column, num_chunks),
                desc=text_column.name,
            )):
        stanford_input_file = os.path.join(working_dir, f'{idx}.txt')
        chunk.to_csv(
            stanford_input_file,
            index=False,
            header=False,
        )

        process_args = JAVA_PARAMS + [
            stanford_input_file, '-outputDirectory', working_dir
        ]
        subprocess.run(process_args, capture_output=not verbose)


def read_results(working_dir):
    def numeric_sort(x):
        return int(x.split('.')[0])

    files = sorted(os.listdir(working_dir), key=numeric_sort)
    files = filter(lambda x: x.endswith('.json'), files)

    for f in tqdm(files, desc='Merging results'):
        with open(os.path.join(working_dir, f)) as in_f:
            payload = json.load(in_f)

            for s in payload['sentences']:
                yield s['index'], [x['word'] for x in s['tokens']]


if __name__ == '__main__':
    main()
