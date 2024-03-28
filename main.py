#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders
from predict import validate
from utils.predict_utils import dynamic_evaluate
from train import train
from utils.utils import save_checkpoint, load_checkpoint, measure_flops, load_state_dict

# import sys
# if all(['models' not in sys.path]):
#     sys.path.extend([f'{sys.path[0]}/models', f'{sys.path[0]}/utils', f'{sys.path[0]}/data_tools'])

np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)


def main():
    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    config = Config()

    measure_flops(args, {**config.model_params[args.data][args.arch]})
    model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
    args.num_exits = config.model_params[args.data][args.arch]['num_blocks']
    args.inference_params = config.inference_params[args.data][args.arch]

    if args.use_gpu:
        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    base_params = [v for k, v in model.named_parameters() if 'ee' not in k]
    exit_params = [v for k, v in model.named_parameters() if 'ee' in k]

    optimizer = torch.optim.SGD([{'params': base_params},
                                 {'params': exit_params}],
                                lr=config.training_params[args.data][args.arch]['lr'],
                                momentum=config.training_params[args.data][args.arch]['momentum'],
                                weight_decay=config.training_params[args.data][args.arch]['weight_decay'])

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    batch_size = args.batch_size if args.batch_size else config.training_params[args.data][args.arch]['batch_size']
    train_loader, val_loader, test_loader = get_dataloaders(args, batch_size)

    if args.evalmode is not None:
        load_state_dict(args, model)
        if args.evalmode == 'anytime':
            validate(model, test_loader, criterion, args)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1\tval_prec1\ttrain_prec5\tval_prec5']

    for epoch in range(args.start_epoch, config.training_params[args.data][args.arch]['num_epoch']):

        train_loss, train_prec1, train_prec5, lr, _, _ = \
            train(model, train_loader, criterion, optimizer, epoch, args, config.training_params[args.data][args.arch])

        val_loss, val_prec1, val_prec5, val_prec1_per_exit, val_prec5_per_exit = validate(model, val_loader, criterion,
                                                                                          args)

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print('Best var_prec1 {}'.format(best_prec1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores, val_prec1_per_exit, val_prec5_per_exit)

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    ### Test the final model
    print('********** Final prediction results **********')
    validate(model, test_loader, criterion, args)

    return


if __name__ == '__main__':
    main()
