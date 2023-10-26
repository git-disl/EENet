from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import pickle as pkl
import time

import pandas as pd

from predict_helpers import *


def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save_path, 'logits_single.pth')):
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save_path, 'logits_single.pth'))
    else:
        val_pred, val_target, val_time = tester.calc_logit(val_loader)
        test_pred, test_target, test_time = tester.calc_logit(test_loader)
        torch.save((val_pred, val_target, test_pred, test_target),
                   os.path.join(args.save_path, 'logits_single.pth'))

    flops = torch.load(os.path.join(args.save_path, 'flops.pth'))
    if not flops:
        flops = [1, 2, 3, 4]
    seconds = pd.read_csv(os.path.join(args.save_path, 'seconds.csv'), header=None)

    if args.use_gpu:
        seconds = list(seconds.iloc[:, 0])
        costs = torch.tensor(seconds).cuda()
        val_pred = val_pred.cuda()
        val_target = val_target.cuda()
        test_pred = test_pred.cuda()
        test_target = test_target.cuda()
    else:
        seconds = list(seconds.iloc[:, 0])
        costs = torch.tensor(seconds)

    n_stage, n_sample, c = val_pred.size()

    nn_array = torch.zeros((n_stage, n_sample))
    test_nn_array = torch.zeros((n_stage, test_pred.shape[1]))

    result_dir = os.path.join(args.save_path,
                              f'{args.inference_save_filename}_{args.exit_distribution_method}_{args.conf_mode}.txt')

    if args.exit_distribution_method == 'exp':

        if args.conf_mode == 'nn':
            args.inference_params['alpha_cost'] = 0
            args.inference_params['alpha_ce'] = 0
            _, _, ea = run_exit_assigner(args, val_pred, val_target, costs, seconds[-1])

        for p in np.linspace(1, 81, 80):
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 15)
            probs = torch.exp(torch.log(_p) * torch.arange(1, args.num_exits + 1)).numpy()
            probs /= probs.sum()
            probs = probs.tolist()
            cost = sum([probs[i] * seconds[i] for i in range(len(probs))])
            if args.val_budget is not None and cost > args.val_budget:
                break

            if args.val_budget is None:
                acc_val, _, _, T = \
                    tester.dynamic_eval_find_threshold(val_pred, val_target, probs, flops, seconds,
                                                       conf_mode=args.conf_mode, nn_array=nn_array)

                handle_eval_outputs(tester, test_pred, test_target, test_nn_array, probs, flops, seconds, T,
                                    test_loader, args, result_dir, None, None)

        if args.val_budget is not None:
            acc_val, _, _, T = \
                tester.dynamic_eval_find_threshold(val_pred, val_target, probs, flops, seconds,
                                                   conf_mode=args.conf_mode, nn_array=nn_array)

            handle_eval_outputs(tester, test_pred, test_target, test_nn_array, probs, flops, seconds, T,
                                test_loader, args, result_dir, None, None)

    elif args.exit_distribution_method == 'nn':

        if args.val_budget is not None:
            budget_list = np.linspace(args.val_budget, args.val_budget + 1, 1)
        else:
            budget_list = np.linspace(seconds[0], seconds[-1], 20)

        for budget in budget_list:

            try:
                ea = pkl.load(open(f'{args.save_path}/ea_pkls/ea_{budget}.pkl', 'rb'))
                probs = pkl.load(open(f'{args.save_path}/ea_pkls/probs_{budget}.pkl', 'rb'))
                find_flag = True
                num_trial = 1
            except:
                ea = None
                probs = None
                find_flag = False
                num_trial = 10

            best_acc_val = 0
            best_package = None
            for trial_idx in range(num_trial):
                print(f'{trial_idx + 1}/{num_trial}')
                if find_flag:
                    T = ea.get_threshold().detach().cpu().numpy()
                else:
                    T, probs, ea = run_exit_assigner(args, val_pred, val_target, costs, budget)

                nn_array = test_exit_assigner(args, val_pred, n_stage, ea)
                test_nn_array = test_exit_assigner(args, test_pred, n_stage, ea)

                acc_val, _, _, _, _, _ = tester.dynamic_eval_with_threshold(val_pred, val_target, flops,
                                                                            seconds, T,
                                                                            conf_mode=args.conf_mode,
                                                                            nn_array=nn_array)

                cost = sum([probs[i] * seconds[i] for i in range(args.num_exits)])
                if acc_val > best_acc_val and np.abs(cost - args.val_budget) / args.val_budget < 0.1:
                    best_acc_val = acc_val
                    best_package = (nn_array, test_nn_array, T, probs, ea, acc_val)

            if best_package is not None:
                nn_array, test_nn_array, T, probs, ea, acc_val = best_package

            if not find_flag:
                pkl.dump(ea, open(f'{args.save_path}/ea_pkls/ea_{budget}_.pkl', 'wb'))
                pkl.dump(probs, open(f'{args.save_path}/ea_pkls/probs_{budget}_.pkl', 'wb'))

            handle_eval_outputs(tester, test_pred, test_target, test_nn_array, probs, flops, seconds, T, test_loader,
                                args, result_dir, None, None)

    else:
        raise NotImplementedError


def test_exit_assigner(args, pred, n_stage, ea):
    nn_array = torch.zeros((n_stage, pred.shape[1]))
    for k in range(n_stage):
        with torch.no_grad():
            X, _ = prepare_input(pred[:k + 1], k=k)
            if k > 0:
                X = torch.concat([X.cpu(), nn_array[:k].permute(1, 0)], dim=-1)
            if args.use_gpu:
                X = X.cuda()
            nn_array[k] = ea.score_normalizers[k].predict(X)[:, 0].cpu()

    return nn_array


def run_exit_assigner(args, val_pred, val_target, costs, budget):
    weight_decay = args.inference_params['weight_decay']
    beta_ce = args.inference_params['beta_ce']
    alpha_ce = args.inference_params['alpha_ce']
    alpha_cost = args.inference_params['alpha_cost']
    lr = args.inference_params['lr']
    num_epoch = args.inference_params['num_epoch']
    batch_size = args.inference_params['bs']
    hidden_dim_rate = args.inference_params['hidden_dim_rate']
    period = args.inference_params['period']

    ea, probs = fit_exit_assigner(val_pred, val_target, costs, budget, alpha_ce=alpha_ce, alpha_cost=alpha_cost,
                                  beta_thr=0, beta_ce=beta_ce, lr=lr, weight_decay=weight_decay, num_epoch=num_epoch, batch_size=batch_size,
                                  hidden_dim_rate=hidden_dim_rate, period=period, conf_mode=args.conf_mode)
    T = ea.get_threshold().detach().cpu().numpy()

    if not isinstance(probs, list):
        probs = probs.tolist()

    return T, probs, ea


def handle_eval_outputs(tester, test_pred, test_target, test_nn_array, probs, flops, seconds, T, test_loader, args, result_dir, package_dict, budget):
    conf_mode = args.conf_mode
    acc_test, exp_flops, exp_seconds, _, exited_samples, probs_ = tester.dynamic_eval_with_threshold(test_pred, test_target,
                                                                                             flops, seconds, T,
                                                                                             conf_mode=conf_mode,
                                                                                             nn_array=test_nn_array)

    print('accuracy: {:.3f}, test flops: {:.2f}M, probs: {}, probs: {}, time: {}'.format(acc_test,
                                                                                          exp_flops / 1e6,
                                                                                          probs, probs_,
                                                                                          exp_seconds))

    with open(result_dir, 'a') as fout:
        fout.write('{}\t{}\t{}\t{}\n'.format(probs, acc_test, exp_flops, exp_seconds))


def create_exit_count_array(n_c_sample_list, n_stage, p):
    c = len(n_c_sample_list)
    out_array = np.zeros((c, n_stage))

    p_out = np.zeros((c, n_stage))

    for c_ in range(c):
        p_out[c_, :] = p

    for k in range(n_stage):
        for c_ in range(c):
            n_c_sample = n_c_sample_list[c_]
            out_n = round(n_c_sample * p_out[c_, k])
            res_n = out_array[c_, :k].sum()
            out_n = min(out_n, n_c_sample - res_n)
            out_array[c_, k] = out_n

    return out_array


class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        if args is None or args.use_gpu:
            self.softmax = nn.Softmax(dim=1).cuda()
        else:
            self.softmax = nn.Softmax(dim=1)

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = self.args.num_exits
        logits = [[] for _ in range(n_stage)]
        targets = []
        avg_time = []

        torch.manual_seed(0)
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                st = time.time()
                output, _ = self.model(input_var, manual_early_exit_index=0)
                et = time.time()
                avg_time.append(et - st)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t)

            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        avg_time = sum(avg_time[10:]) / (len(avg_time) - 10)
        print(avg_time)

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets, avg_time

    def dynamic_eval_find_threshold(self, logits, targets, p, flops, seconds, conf_mode='maxpred', nn_array=None):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        conf_scores = torch.zeros((n_stage, n_sample))
        for exit_idx in range(n_stage):
            pred = logits[exit_idx]
            pred_log = torch.log(pred)
            conf = 1 + torch.sum(pred * pred_log, dim=1) / math.log(pred.shape[1])
            conf_scores[exit_idx] = conf

        if conf_mode == 'maxpred':
            _, sorted_idx = max_preds.sort(dim=1, descending=True)
            criteria_vals = max_preds
        elif conf_mode == 'nn':
            val = nn_array
            _, sorted_idx = val.sort(dim=1, descending=True)
            criteria_vals = val
        elif conf_mode == 'vote':
            criteria_vals = np.stack([np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[1][-1] / (k + 1),
                                                          axis=0, arr=argmax_preds[:k + 1].cpu().numpy())
                                      for k in range(n_stage)])
            criteria_vals = torch.tensor(criteria_vals + np.random.randn(*criteria_vals.shape) * 1e-6)
            _, sorted_idx = criteria_vals.sort(dim=1, descending=True)
        elif conf_mode == 'entropy':
            _, sorted_idx = conf_scores.sort(dim=1, descending=True)
            criteria_vals = conf_scores
        else:
            raise NotImplementedError

        T = ExitAssigner.compute_threshold(criteria_vals.permute(1, 0), p)

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops, expected_seconds = 0, 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                threshold = T[k]
                if criteria_vals[k][i].item() >= threshold:  # force to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            expected_seconds += _t * seconds[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, expected_seconds, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, seconds, T, conf_mode='maxpred', nn_array=None):
        n_stage, n_sample, c = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        conf_scores = torch.zeros((n_stage, n_sample))
        for exit_idx in range(n_stage):
            pred = logits[exit_idx]
            pred_log = torch.log(pred)
            conf = 1 + torch.sum(pred * pred_log, dim=1) / math.log(pred.shape[1])
            conf_scores[exit_idx] = conf

        if conf_mode == 'maxpred':
            _, sorted_idx = max_preds.sort(dim=1, descending=True)
            criteria_vals = max_preds
        elif conf_mode == 'nn':
            val = nn_array
            _, sorted_idx = val.sort(dim=1, descending=True)
            criteria_vals = val
        elif conf_mode == 'entropy':
            _, sorted_idx = conf_scores.sort(dim=1, descending=True)
            criteria_vals = conf_scores
        else:
            raise NotImplementedError

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops, expected_seconds = 0, 0, 0
        exited_samples = [[] for _ in range(n_stage)]
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                threshold = T[k]
                if criteria_vals[k][i].item() >= threshold:  # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    exited_samples[k].append(i)
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break

        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            expected_seconds += _t * seconds[k]
            acc_all += acc_rec[k]

        # print('test:', exp / sum(exp))

        exit_accuracies = [(acc_rec[i] / exp[i]).item() if exp[i] else .5 for i in range(n_stage)]

        return acc * 100.0 / n_sample, expected_flops, expected_seconds, exit_accuracies, exited_samples, exp / n_sample
