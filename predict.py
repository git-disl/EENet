#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
import torch.nn.parallel
import torch.optim

from utils import accuracy, AverageMeter


def validate(model, val_loader, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.num_exits):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_gpu:
                target = target.cuda()
                input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output, _ = model(input_var, conf_early_exit=False)
            if not isinstance(output, list):
                output = [output]

            loss = torch.zeros(0)
            for j in range(len(output)):
                if 'bert' in model.__class__.__name__:
                    loss += (j + 1) * criterion(output[j], target_var) / (args.num_exits * (args.num_exits + 1))
                else:
                    loss += criterion(output[j], target_var) / args.num_exits

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    for j in range(args.num_exits):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    return losses.avg, top1[-1].avg, top5[-1].avg, top1, top5
