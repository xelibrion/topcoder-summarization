#!/usr/bin/env python
# coding: utf-8

import os
import random
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from nnet import Summarizer
from nnet.meters import AverageMeter

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


def train_epoch(model, iterator, optimizer, criterion, clip):

    loss_meter = AverageMeter(int(len(iterator) / 10))
    model.train()

    epoch_loss = 0

    with tqdm(total=len(iterator)) as tq:
        tq.set_description('Train')

        for batch in iterator:
            optimizer.zero_grad()

            src, text_lengths = batch.text
            trg = src

            output = model(src, text_lengths, trg)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            loss_meter.update(loss.item())
            tq.set_postfix(
                loss='{:.3f}'.format(loss_meter.mavg),
                PPL='{:.2f}'.format(math.exp(loss_meter.mavg)),
            )
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


def load_data(dataset_path, field):
    fields = {'text': ('text', field)}

    print(f"Loading data from '{dataset_path}'")

    train_data, valid_data, test_data = TabularDataset.splits(path=dataset_path,
                                                              train='train.jsonl',
                                                              validation='val.jsonl',
                                                              test='test.jsonl',
                                                              format='json',
                                                              fields=fields)

    print(vars(train_data.examples[0]))

    return train_data, valid_data, test_data


def compose_model():
    model = Summarizer()
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model, nn.CrossEntropyLoss()


def train(dataset_path, resume, lr):
    TEXT = Field(
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
    )

    train_data, valid_data, test_data = load_data(dataset_path, TEXT)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=h.BATCH_SIZE,
        device=device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
    )

    model, criterion = compose_model()
    if resume:
        model.load_state_dict(torch.load(MODEL_PATH))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    N_EPOCHS = 30

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, criterion, h.CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}\n')


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

    args = parser.parse_args()

    dataset = '../data/reviews_short' if args.sample else '../data/reviews'
    train(rel_path(dataset), args.resume, args.lr)


if __name__ == '__main__':
    main()
