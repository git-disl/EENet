#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
import torch.nn.parallel
import torch.optim

from utils import adjust_learning_rate, accuracy, AverageMeter


def train(model, train_loader, criterion, optimizer, epoch, args, train_params):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.num_exits):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()
    end = time.time()

    running_lr = None
    avg_time = []

    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, train_params, batch=i, nBatch=len(train_loader))

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        if args.use_gpu:
            target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        st = time.time()
        output = model(input_var, epoch=epoch)

        et = time.time()
        avg_time.append(et - st)

        if not isinstance(output, list):
            output = [output]

        loss = torch.tensor(0)
        for j in range(len(output)):
            loss += (j + 1) * criterion(output[j], target_var) / (args.num_exits * (args.num_exits + 1))
            if epoch > train_params['num_epoch'] * 0.75 and j < len(output) - 1:
                T = 3
                alpha_kl = 0.01
                loss += torch.nn.KLDivLoss()(torch.log_softmax(output[j] / T, dim=-1),  torch.softmax(output[-1] / T, dim=-1)) * alpha_kl * T * T

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t\t'
                  'Exit: {3}\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.4f}\t'
                  'Acc@5 {top5.val:.4f}'.format(
                epoch, i + 1, len(train_loader), len(output),
                batch_time=batch_time, data_time=data_time,
                loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr, top1, top5
