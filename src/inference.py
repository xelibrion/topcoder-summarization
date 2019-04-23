#!/usr/bin/env python

import argparse
from tqdm import tqdm


def first3_sentences(input_fname):
    with open(input_fname) as fin:
        for line in fin.readlines():
            sentences = line.strip().split('.')
            if len(sentences) > 3:
                sentences = sentences[:3]
            yield ' '.join(['<s> %s .</s>' % s for s in sentences])


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
