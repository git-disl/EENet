import math
import numpy as np
import os
import shutil

import models
import torch

from utils.op_counter import measure_model


def save_checkpoint(state, args, is_best, filename, result, prec1_per_exit, prec5_per_exit):
    print(args)
    result_filename = os.path.join(args.save_path, 'scores.tsv')
    exit_result_filename = os.path.join(args.save_path, 'exit_scores.tsv')
    model_dir = os.path.join(args.save_path, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(exit_result_filename, 'a') as f:
        text = '\t'.join([str(state['epoch'])] + [f'{score.avg:.2f}' for score in prec1_per_exit + prec5_per_exit])
        print(text, file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args):
    model_dir = os.path.join(args.save_path, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, args, params, batch=None, nBatch=None):
    if params['lr_type'] == 'cosine':
        T_total = params['num_epoch'] * nBatch
        T_cur = (epoch % params['num_epoch']) * nBatch + batch
        lr = 0.5 * params['lr'] * (1 + math.cos(math.pi * T_cur / T_total))
    elif params['lr_type'] == 'multistep':
        if args.data.startswith('cifar') or args.data == 'tinyimagenet':
            lr, decay_rate = params['lr'], 0.1
            if epoch >= params['decay_epochs'][1]:
                lr *= decay_rate ** 2
            elif epoch >= params['decay_epochs'][0]:
                lr *= decay_rate
        else:
            lr = params['lr'] * (0.1 ** (epoch // 30))
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr
    return lr


def adjust_exit_learning_rate(optimizer, epoch, args, params, batch=None, nBatch=None):
    if params['lr_type'] == 'cosine':
        T_total = params['num_epochs'] * nBatch
        T_cur = (epoch % params['num_epochs']) * nBatch + batch
        lr = 0.5 * params['lr'] * (1 + math.cos(math.pi * T_cur / T_total))
    elif params['lr_type'] == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = params['lr'], 0.1
            if epoch >= params['decay_epochs'][1]:
                lr *= decay_rate ** 2
            elif epoch >= params['decay_epochs'][0]:
                lr *= decay_rate
        else:
            lr = params['lr'] * (0.1 ** ((epoch - 200) // 20))
    optimizer.param_groups[1]['lr'] = lr
    return lr


def update_confidence_scores(old_conf_scores, score_order, new_conf_scores, start_idx, alpha=0.9):
    end_idx = start_idx + len(new_conf_scores)
    old_conf_scores = np.array(old_conf_scores, dtype=float)
    old_conf_scores[score_order[start_idx: end_idx]] = np.array(new_conf_scores) * alpha + \
                                                       old_conf_scores[score_order[start_idx: end_idx]] * (1 - alpha)
    return old_conf_scores.tolist()


def measure_flops(args, params):
    model = getattr(models, args.arch)(args, params)
    model.eval()
    n_flops, n_params = measure_model(model, args.image_size[0], args.image_size[1], exit_idx=4)
    torch.save(n_flops, os.path.join(args.save_path, 'flops.pth'))
    del (model)


def load_state_dict(args, model):
    if args.use_gpu:
        state_dict = torch.load(args.evaluate_from)['state_dict']
    else:
        state_dict = torch.load(args.evaluate_from, map_location='cpu')['state_dict']

    if not args.use_gpu:
        state_dict_ = {}
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                state_dict_[k[7:]] = v
            else:
                state_dict_[k] = v
    else:
        state_dict_ = state_dict

    # if 'bert' in args.arch:
    #     state_dict_ = {}
    #     for k, v in state_dict.items():
    #         state_dict_['module.'+k] = v
    # else:
    # state_dict_ = state_dict

    model.load_state_dict(state_dict_, strict=False)
