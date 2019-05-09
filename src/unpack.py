#!/usr/bin/env python
import os
import struct
import json
from tqdm import tqdm


def unpack_strings(file_path):
    with open(file_path, 'rb') as reader:
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_str.decode('utf-8', 'ignore')


def split_abstract_body(unpacked_string):
    abstract_start = unpacked_string.index('<s>')
    abstract_end = unpacked_string.rindex('</s>') + len('</s>')
    abstract = unpacked_string[abstract_start:abstract_end]
    article_start = unpacked_string[abstract_end:].index('article') + len('article')
    article = unpacked_string[abstract_end + article_start:]
    return abstract, article


def main():
    current_file = os.path.abspath(__file__)
    src_file = os.path.join(os.path.dirname(current_file), '../data/train.bin')
    dest_file = os.path.join(os.path.dirname(current_file), '../data/unpacked.jsonl')

    with open(dest_file, 'w') as f:
        for idx, item in enumerate(tqdm(unpack_strings(src_file))):
            abstract, article = split_abstract_body(item)
            parsed = {
                'article_id': idx,
                'abstract': abstract,
                'article': article,
            }
            f.write(json.dumps(parsed))
            f.write('\n')


if __name__ == '__main__':
    main()
