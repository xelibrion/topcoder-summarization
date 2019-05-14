#!/usr/bin/env python
# coding: utf-8

import os
import random
import argparse

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule, BertAdam
from apex.optimizers import FP16_Optimizer
from apex.optimizers import FusedAdam

from nnet import Summarizer
from nnet.dataset import SentencesDataset
from training_loop import TrainigLoop, LearningRateScheduler, DummyLRateScheduler

MODEL_PATH = 'summarizer_model.pt'

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_optimizer(model_params,
                    learning_rate,
                    warmup_proportion,
                    total_train_steps,
                    use_fp16=False):
    def pop_layers(params, name_filter):
        params_dict = dict(params)
        layer_names = [n for n, _ in params if not any(nf in n for nf in name_filter)]
        layers = []
        for name in layer_names:
            layers.append(params_dict.pop(name))

        return list(params_dict.items()), layers

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in model_params if 'pooler' not in n[0]]

    params, custom_layers = pop_layers(param_optimizer, ['decoder'])
    params, no_decay_layers = pop_layers(params,
                                         ['bias', 'LayerNorm.bias', 'LayerNorm.weight'])
    the_rest_layers = [p for _, p in params]

    optimizer_grouped_parameters = [{
        'params': custom_layers,
        'weight_decay': 0.01,
        'lr': 1e-3
    }, {
        'params': the_rest_layers,
        'weight_decay': 0.01
    }, {
        'params': no_decay_layers,
        'weight_decay': 0.0
    }]

    if use_fp16:
        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        warmup_linear = WarmupLinearSchedule(
            warmup=warmup_proportion,
            t_total=total_train_steps,
        )
        return optimizer, LearningRateScheduler(warmup_linear, learning_rate)

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        warmup=warmup_proportion,
        t_total=total_train_steps,
    )
    return optimizer, DummyLRateScheduler()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compose_model():
    model = Summarizer()
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model.to(device), nn.BCELoss(reduction='none')


def load_data(dataset_path, batch_size, train_steps, eval_steps):
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
        SentencesDataset(train.values, train_steps * batch_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )

    val_iterator = DataLoader(
        SentencesDataset(val.values, eval_steps * batch_size),
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


def train(
        dataset_path,
        resume,
        lr,
        batch_size,
        epochs=50,
        train_steps=2000,
        eval_steps=500,
):
    train_iterator, val_iterator = load_data(
        dataset_path,
        batch_size,
        train_steps=train_steps,
        eval_steps=eval_steps,
    )

    model, _ = compose_model()
    if resume:
        model.load_state_dict(torch.load(MODEL_PATH))

    model_params = list(model.named_parameters())
    # for _, p in model_params[:-4]:
    #     p.requires_grad = False

    # params_to_train = model_params[-4:]
    params_to_train = model_params

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer, lr_scheduler = build_optimizer(
        params_to_train,
        lr,
        0.01,
        epochs * train_steps,
    )

    loop = TrainigLoop(
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        callbacks={'on_best_val_loss': save_checkpoint},
    )
    loop.run(train_iterator, val_iterator, epochs)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=7,
    )

    args = parser.parse_args()
    print(args)

    dataset = '../data/ready'
    train(
        rel_path(dataset),
        args.resume,
        args.lr,
        args.batch_size,
    )


if __name__ == '__main__':
    main()
