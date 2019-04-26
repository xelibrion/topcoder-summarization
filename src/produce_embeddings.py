#!/usr/bin/env python
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.externals import joblib

BERT_CACHE = '~/.pytorch_pretrained_bert'
BERT_VOCAB = '26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084'
BERT_MODEL = '9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba'


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def pad(token_ids, expected_length):
    to_pad = expected_length - len(token_ids)
    return token_ids + [0] * to_pad


def prepare_batch_item(sentence, max_sequence_size=512):
    tokens = tokenizer.tokenize(sentence)
    input_ids_list = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    mask = [1] * len(input_ids_list)
    input_ids_list = pad(input_ids_list, max_sequence_size)
    mask = pad(mask, max_sequence_size)

    return torch.Tensor(input_ids_list).long(), torch.Tensor(mask).long()


class SentencesDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return self.sentences.shape[0]

    def __getitem__(self, idx):
        row = self.sentences.iloc[idx]
        metadata = torch.Tensor([row['article_id'], row['group_id']]).long()
        return metadata, prepare_batch_item(row['sentence'])


device = torch.device('cuda')


def bert_path(filename):
    p = os.path.join(BERT_CACHE, filename)
    return os.path.expanduser(p)


tokenizer = BertTokenizer.from_pretrained(bert_path(BERT_VOCAB))
model = BertModel.from_pretrained(bert_path(BERT_MODEL)).to(device)

df = pd.read_json(
    rel_path('../data/sentences_grouped.jsonl'),
    lines=True,
    orient='records',
)

meta_all = []
embeddings_all = []
pooled_all = []

BATCH_SIZE = 64
NUM_LAYERS_TO_KEEP = 4

batcher = DataLoader(
    SentencesDataset(df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=12,
    pin_memory=True,
)

EMBEDDINGS_DIR = rel_path('../data/embeddings/')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

with torch.no_grad():
    model.eval()

    with tqdm(total=len(batcher.dataset)) as tq:
        for batch_idx, (meta, inputs) in enumerate(batcher):
            input_ids, mask = inputs
            input_ids, mask = input_ids.to(device), mask.to(device)

            encoder_layers, pooled = model(input_ids,
                                           attention_mask=mask,
                                           output_all_encoded_layers=True)

            embeddings_l = encoder_layers[-NUM_LAYERS_TO_KEEP:]
            embeddings = torch.cat([t.unsqueeze(1) for t in embeddings_l], 1)

            tq.update(meta.size(0))
            meta_all.append(meta)
            pooled_all.append(pooled.cpu())
            embeddings_all.append(embeddings.cpu())

            if batch_idx % 10 == 0:
                joblib.dump(torch.cat(meta_all),
                            rel_path(f'../data/embeddings/{batch_idx}_meta.pkl'),
                            compress=9)
                joblib.dump(torch.cat(pooled_all),
                            rel_path(f'../data/embeddings/{batch_idx}_pooled.pkl'),
                            compress=9)
                joblib.dump(torch.cat(embeddings_all),
                            rel_path(f'../data/embeddings/{batch_idx}_embeddings.pkl'),
                            compress=9)
                meta_all = []
                embeddings_all = []
                pooled_all = []

joblib.dump(torch.cat(meta_all), rel_path(f'../data/embeddings/{batch_idx}_meta.pkl'))
joblib.dump(torch.cat(pooled_all), rel_path(f'../data/embeddings/{batch_idx}_pooled.pkl'))
joblib.dump(torch.cat(embeddings_all),
            rel_path(f'../data/embeddings/{batch_idx}_embeddings.pkl'))
