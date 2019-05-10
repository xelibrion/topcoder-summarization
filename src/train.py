#!/usr/bin/env python
# coding: utf-8

import os
import random
import math
import time
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from nnet import Summarizer
from nnet.meters import AverageMeter
from nnet.dataset import SentencesDataset

MODEL_PATH = 'summarizer_model.pt'

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, iterator, optimizer, criterion):

    loss_meter = AverageMeter(int(len(iterator) / 10))
    model.train()

    epoch_loss = 0

    with tqdm(total=len(iterator)) as tq:
        tq.set_description('Train')

        for batch in iterator:
            optimizer.zero_grad()

            for idx, t in enumerate(batch):
                batch[idx] = t.to(device)

            input_ids, attention_mask, segments_ids, cls_ids, cls_mask, labels = batch
            # labels = [batch_size x 512]

            output = model(input_ids, attention_mask, segments_ids, cls_ids, cls_mask)
            sent_scores, out_cls_mask = output

            # sent_scores = [batch_size x 512]
            # out_cls_mask = [batch_size x 512]

            loss = criterion(sent_scores, labels)
            loss = (loss * out_cls_mask.float()).sum()
            (loss / loss.numel()).backward()

            optimizer.step()

            loss_meter.update(loss.item())
            tq.set_postfix(loss='{:.3f}'.format(loss_meter.mavg), )
            tq.update()

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()
    loss_meter = AverageMeter(int(len(iterator) / 10))

    epoch_loss = 0

    with tqdm(total=len(iterator)) as tq:
        tq.set_description('Validation')

        with torch.no_grad():

            for batch in iterator:
                for idx, t in enumerate(batch):
                    batch[idx] = t.to(device)

                input_ids, attention_mask, segments_ids, cls_ids, cls_mask, labels = batch
                # labels = [batch_size x 512]

                output = model(input_ids, attention_mask, segments_ids, cls_ids, cls_mask)
                sent_scores, out_cls_mask = output

                loss = criterion(sent_scores, labels)
                loss = (loss * out_cls_mask.float()).sum()

                epoch_loss += loss.item()
                loss_meter.update(loss.item())
                tq.set_postfix(loss='{:.3f}'.format(loss_meter.mavg), )
                tq.update()

    return epoch_loss / len(iterator)


def compose_model():
    model = Summarizer()
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model.to(device), nn.BCELoss(reduction='none')


def load_data(dataset_path, batch_size):
    train = pd.read_json(
        os.path.join(dataset_path, 'train.jsonl'),
        lines=True,
        orient='records',
    )
    val = pd.read_json(
        os.path.join(dataset_path, 'val.jsonl'),
        lines=True,
        orient='records',
    )

    train_iterator = DataLoader(
        SentencesDataset(train.values),
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    val_iterator = DataLoader(
        SentencesDataset(val.values),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_iterator, val_iterator


def train(dataset_path, resume, lr, batch_size):

    train_iterator, val_iterator = load_data(dataset_path, batch_size)

    model, criterion = compose_model()
    if resume:
        model.load_state_dict(torch.load(MODEL_PATH))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    N_EPOCHS = 30

    best_val_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, criterion)
        val_loss = evaluate(model, val_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}\n')


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--sample',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
    )

    args = parser.parse_args()

    dataset = '../data/ready'
    train(
        rel_path(dataset),
        args.resume,
        args.lr,
        args.batch_size,
    )


if __name__ == '__main__':
    main()
