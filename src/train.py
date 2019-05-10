#!/usr/bin/env python
# coding: utf-8

import os
import random
import math
import time
import argparse

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from nnet import Summarizer
from nnet.meters import AverageMeter
from nnet.dataset import SentencesDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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

            output = model(input_ids, attention_mask, segments_ids, cls_ids, cls_mask)
            sent_scores, out_cls_mask = output

            loss = criterion(sent_scores, labels)
            loss = (loss * out_cls_mask.float()).sum()
            (loss / loss.numel()).backward()

            # loss = self.loss(sent_scores, labels.float())
            # loss = (loss*mask.float()).sum()
            # (loss/loss.numel()).backward()

            optimizer.step()

            loss_meter.update(loss.item())
            tq.set_postfix(loss='{:.3f}'.format(loss_meter.mavg), )
            tq.update()

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with tqdm(total=len(iterator)) as tq:
        tq.set_description('Validation')

        with torch.no_grad():

            for batch in iterator:
                src, text_lengths = batch.text
                trg = src

                output = model(src, text_lengths, trg, 0)  # turn off teacher forcing

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]

                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

                # trg = [(trg sent len - 1) * batch size]
                # output = [(trg sent len - 1) * batch size, output dim]

                loss = criterion(output, trg)

                epoch_loss += loss.item()
                tq.update()

    return epoch_loss / len(iterator)


def compose_model():
    model = Summarizer()
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model.to(device), nn.BCELoss(reduction='none')


def load_data(dataset_path, batch_size):
    train_text = pd.read_json(
        os.path.join(dataset_path, 'train_text.jsonl'),
        lines=True,
        orient='records',
    )
    train_oracle = pd.read_json(
        os.path.join(dataset_path, 'train_oracle.jsonl'),
        lines=True,
        orient='records',
    )

    val_text = pd.read_json(
        os.path.join(dataset_path, 'val_text.jsonl'),
        lines=True,
        orient='records',
    )
    val_oracle = pd.read_json(
        os.path.join(dataset_path, 'val_oracle.jsonl'),
        lines=True,
        orient='records',
    )

    train_text, train_oracle = train_text.head(1000), train_oracle.head(1000)
    val_text, val_oracle = val_text.head(1000), val_oracle.head(1000)

    train_iterator = DataLoader(
        SentencesDataset(
            train_text['abstract_sentences'].tolist(),
            train_text['article_sentences'].tolist(),
            train_oracle['selected_sentences_ids'].tolist(),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    val_iterator = DataLoader(
        SentencesDataset(
            val_text['abstract_sentences'].tolist(),
            val_text['article_sentences'].tolist(),
            val_oracle['selected_sentences_ids'].tolist(),
        ),
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
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n')


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
        default=12,
    )

    args = parser.parse_args()

    dataset = '../train_data'
    train(
        rel_path(dataset),
        args.resume,
        args.lr,
        args.batch_size,
    )


if __name__ == '__main__':
    main()
