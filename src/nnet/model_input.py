import numpy as np
from typing import List

from pytorch_pretrained_bert import BertTokenizer


class InputProcessor():
    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']
        self.min_src_ntokens = 5
        self.max_src_ntokens = 200
        self.min_nsents = 3
        self.max_nsents = 100

    def preprocess(
            self,
            src: List[List[str]],
            tgt: List[List[str]],
            oracle_ids: List[int],
    ):

        if not src:
            return None

        orig_sentences = np.array([' '.join(s) for s in src])

        labels = np.zeros(len(src))
        labels[oracle_ids] = 1

        idx_long_enough = [i for i, s in enumerate(src) if len(s) > self.min_src_ntokens]

        src_filtered = [src[i][:self.max_src_ntokens] for i in idx_long_enough]
        labels = labels[idx_long_enough]
        src_filtered = src_filtered[:self.max_nsents]
        labels = labels[:self.max_nsents]

        if len(src) < self.min_nsents:
            return None
        if not np.any(labels):
            return None

        src_filtered_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_filtered_txt)
        src_tokens = self.tokenizer.tokenize(text)

        src_tokens = src_tokens[:510]
        src_tokens = ['[CLS]'] + src_tokens + ['[SEP]']

        src_token_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        src_token_ids = np.array(src_token_ids)

        sep_ids = np.where(src_token_ids == self.sep_vid)[0]
        cls_ids = np.where(src_token_ids == self.cls_vid)[0]

        segments = np.zeros_like(src_token_ids)
        segment_lo = np.array([-1] + sep_ids.tolist()) + 1
        for idx, (lo, hi) in enumerate(zip(segment_lo, sep_ids)):
            if idx % 2 != 0:
                np.put(segments, range(lo, hi + 1), 1)

        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = orig_sentences[idx_long_enough]
        return src_token_ids, labels, segments, cls_ids, src_txt, tgt_txt
