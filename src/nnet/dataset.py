import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

from .model_input import InputProcessor

import spacy


def pad(token_ids, expected_length):
    to_pad = expected_length - len(token_ids)
    return token_ids + [0] * to_pad


def prepare_batch_item(tokenizer, sentence, max_sequence_size=512):
    tokens = tokenizer.tokenize(sentence)
    input_ids_list = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    mask = [1] * len(input_ids_list)
    input_ids_list = pad(input_ids_list, max_sequence_size)
    mask = pad(mask, max_sequence_size)

    return torch.Tensor(input_ids_list).long(), torch.Tensor(mask).long()


class SentencesDataset(Dataset):
    def __init__(self, abstracts, articles, oracle_ids):
        # ["abstract_sentences", "article_id", "article_sentences"]

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.nlp = spacy.load('en_core_web_lg')
        # TODO: proper tokens
        abstracts_tokens = [[s.split(' ') for s in ss] for ss in abstracts]
        articles_tokens = [[s.split(' ') for s in ss] for ss in articles]

        self.data = []
        for src, tgt, or_ids in tqdm(zip(articles_tokens, abstracts_tokens, oracle_ids),
                                     'Preprocessing'):
            result = InputProcessor().preprocess(
                src,
                tgt,
                or_ids,
            )
            if result:
                self.data.append(result)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(self.data[idx])
        return self.data[idx]
