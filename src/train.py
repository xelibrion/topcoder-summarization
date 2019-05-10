#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule
from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam

from nnet import Summarizer
from nnet.dataset import SentencesDataset
from training_loop import TrainigLoop, LearningRateScheduler

MODEL_PATH = 'summarizer_model.pt'

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_optimizer(
        model_params,
        learning_rate,
        warmup_proportion,
        total_train_steps,
):
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in model_params if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    optimizer = FusedAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        bias_correction=False,
        max_grad_norm=1.0,
    )
    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                         t_total=total_train_steps)

    return optimizer, warmup_linear


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        shuffle=True,
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


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def save_checkpoint(model):
    torch.save(model.state_dict(), MODEL_PATH)


def train(dataset_path, resume, lr, batch_size, epochs=50, steps_per_epoch=1000):
    train_iterator, val_iterator = load_data(dataset_path, batch_size)

    model, criterion = compose_model()
    if resume:
        model.load_state_dict(torch.load(MODEL_PATH))

    optimizer, schedule = build_optimizer(
        list(model.named_parameters()),
        lr,
        0.1,
        epochs * steps_per_epoch,
    )

    loop = TrainigLoop(
        model,
        optimizer,
        lr_scheduler=LearningRateScheduler(schedule, lr),
        callbacks={'on_best_val_loss': save_checkpoint},
    )
    loop.run(train_iterator, val_iterator, criterion)


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
        default=5e-5,
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
