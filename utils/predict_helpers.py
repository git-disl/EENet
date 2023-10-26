import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt


class ScoreNormalizer(nn.Module):
    def __init__(self, in_dim, c, ratio, alpha_adj=1):
        super(ScoreNormalizer, self).__init__()
        self.c = c
        self.alpha_adj = alpha_adj

        mid_dim = int(in_dim * ratio)

        self.q_layer_1 = nn.Sequential(nn.Linear(c, 1, bias=False))
        self.q_layer_2 = nn.Sequential(nn.Linear(in_dim - c, 1, bias=False))

        self.prob_out = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.Sigmoid(), nn.Linear(mid_dim, 1), nn.Sigmoid())

    def forward(self, X):
        q_hat = self.q_layer_1(X[:, :self.c]) * self.alpha_adj + self.q_layer_2(X[:, self.c:]).unsqueeze(-1)
        r_ = self.prob_out(X)
        return q_hat, r_

    def predict(self, X):
        q_hat = self.q_layer_1(X[:, :self.c]) * self.alpha_adj + self.q_layer_2(X[:, self.c:]).unsqueeze(-1)
        return q_hat


class ExitAssigner(nn.Module):
    def __init__(self, score_normalizers, costs, budget, alpha_ce, alpha_cost, beta_thr, beta_ce, num_class, num_exit, conf_mode='nn'):
        super(ExitAssigner, self).__init__()
        self.num_class = num_class
        self.num_exit = num_exit
        self.costs = costs
        self.budget = budget
        self.alpha_ce = alpha_ce
        self.alpha_cost = alpha_cost
        self.beta_thr = beta_thr
        self.beta_ce = beta_ce
        self.score_normalizers = nn.ModuleList(score_normalizers)
        self.conf_mode = conf_mode
        self.thresholds = torch.zeros(self.num_exit).to(costs.device)

    def compute_weighted_ce_loss(self, logits, targets, weights):
        loss = - logits + torch.log(torch.exp(logits).sum(dim=1).unsqueeze(-1))
        loss = torch.sum(weights * loss * targets)
        return loss

    def compute_loss(self, logits, targets, X_list, opt_q_flag=True, opt_r_flag=True):
        num_sample = logits.shape[1]
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        score = []
        score_ = []
        prob_score = []
        for k in range(self.num_exit):
            X = X_list[k].clone()
            if k > 0:
                past_scores = torch.stack([score[k_].detach() for k_ in range(k)], dim=-1)
                X = torch.concat([X, past_scores], dim=-1)
            q, prob_out = self.score_normalizers[k].forward(X)
            score_.append(q[:, 0])
            q = torch.clamp(q, 0, 1)
            score.append(q[:, 0])
            prob_score.append(prob_out[:, 0])
        score = torch.stack(score, dim=-1)  # NK
        score_ = torch.stack(score_, dim=-1)  # NK
        prob_score = torch.stack(prob_score, dim=-1)  # NK
        probs = torch.softmax(prob_score, dim=-1)

        # ---------------------------------

        acc_matrix = (argmax_preds == targets).float().permute(1, 0)

        prob_acc_matrix = score.clone().detach() ** self.beta_ce
        prob_acc_matrix = prob_acc_matrix / prob_acc_matrix.sum(dim=-1).unsqueeze(-1)
        # ---------------------------------

        weight = probs.clone().detach()
        bce_weight = weight / weight.sum(dim=0)

        bce_loss = torch.zeros(1).to(bce_weight.device)
        for k in range(self.num_exit):
            bce_loss += nn.BCELoss(weight=bce_weight[:, k], reduction='sum')(score[:, k].flatten(), acc_matrix[:, k].flatten())
        bce_loss /= self.num_exit

        # ---------------------------------

        kl_loss = nn.KLDivLoss()(torch.log(probs), prob_acc_matrix)

        # ---------------------------------

        cost = (probs * self.costs).sum() / num_sample
        cost_loss = torch.abs(cost - self.budget) / self.budget

        loss = bce_loss * opt_q_flag + (self.alpha_ce * kl_loss + self.alpha_cost * cost_loss) * opt_r_flag
        loss_ = bce_loss * opt_q_flag + (self.alpha_ce * kl_loss + self.alpha_cost * cost_loss)

        if self.beta_thr:
            with torch.no_grad():

                if self.conf_mode == 'maxpred':
                    score_ = max_preds.permute(1, 0)
                elif self.conf_mode == 'entropy':
                    conf_scores = torch.zeros_like(score)
                    for exit_idx in range(self.num_exit):
                        pred = logits[exit_idx]
                        pred_log = torch.log(pred)
                        conf = 1 + torch.sum(pred * pred_log, dim=1) / math.log(pred.shape[1])
                        conf_scores[exit_idx] = conf
                    score_ = conf_scores
                elif self.conf_mode == 'vote':
                    criteria_vals = np.stack(
                        [np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[1][-1] / (k + 1),
                                             axis=0, arr=argmax_preds[:k + 1].cpu().numpy())
                         for k in range(self.num_exit)])
                    score_ = torch.tensor(criteria_vals + np.random.randn(*criteria_vals.shape) * 1e-6)

                thr = self.compute_threshold(score_, probs)

                self.thresholds = (1 - self.beta_thr) * self.thresholds + self.beta_thr * thr
        else:
            thr = None

        return loss, thr, probs, (bce_loss, ce_loss, cost_loss), loss_

    def get_threshold(self):
        return self.thresholds

    def get_probs(self, logits, targets, X_list):
        with torch.no_grad():
            _, _, probs, _, _ = self.compute_loss(logits, targets, X_list)
            return probs.mean(dim=0)

    @staticmethod
    def compute_threshold(score, probs):
        num_exit = score.shape[1]
        num_sample = score.shape[0]

        if not isinstance(probs, list):
            probs_mean = probs.mean(dim=0)
        else:
            probs_mean = probs

        num_sample_list = []
        for k in range(num_exit - 1):
            num_sample_list.append(math.floor(probs_mean[k] * num_sample))
        num_sample_list.append(num_sample - sum(num_sample_list))

        thr_list = []
        for k in range(num_exit - 1):
            if num_sample_list[k] == 0:
                thr_list.append(1)
                continue
            sort_idxs = score[:, k].argsort()
            total_sample = num_sample_list[k]
            thr = score[sort_idxs[-total_sample], k]
            thr_list.append(thr)
            score = score[sort_idxs[:-num_sample_list[k]]]

        thr_list.append(0)
        thr_list = torch.tensor(thr_list).to(score.device)
        return thr_list


def fit_exit_assigner(pred, target, costs, budget, alpha_ce, alpha_cost, beta_thr, beta_ce, lr, weight_decay, num_epoch, batch_size,
                      hidden_dim_rate, period, conf_mode):
    num_exit, num_sample, num_class = pred.shape
    score_normalizers = [ScoreNormalizer(num_class + 2 + k, num_class, hidden_dim_rate).to(pred.device) for k in
                         range(num_exit)]
    m = ExitAssigner(score_normalizers, costs, budget, alpha_ce, alpha_cost, beta_thr, beta_ce, num_class, num_exit, conf_mode).to(pred.device)
    optimizer = opt.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)

    X_list = []
    for k in range(num_exit):
        X, _ = prepare_input(pred[:k + 1], k=k)
        X_list.append(X)

    perm = np.random.permutation(num_sample)
    min_loss = 1e3
    best_model = None
    tolerance = 100

    opt_q_flag = True
    opt_r_flag = True

    tol = tolerance
    for epoch_idx in range(num_epoch):

        if period > 1:
            if (epoch_idx // period) % 2 == 0:
                opt_q_flag = True
                opt_r_flag = False
            else:
                opt_q_flag = False
                opt_r_flag = True

        loss_list = []
        loss_tuple_list = []
        for start_idx in np.arange(0, num_sample, batch_size):
            end_idx = start_idx + batch_size
            loss, _, _, loss_tuple, loss_ = m.compute_loss(pred[:, perm[start_idx: end_idx]], target[perm[start_idx: end_idx]],
                                                               [X[perm[start_idx: end_idx]] for X in X_list], opt_q_flag, opt_r_flag)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
            loss_tuple_list.append(loss_tuple)

        avg_loss = sum(loss_list).item() / len(loss_list)
        avg_loss_tuple = [sum([l[i] for l in loss_tuple_list]) / len(loss_tuple_list) for i in range(3)]

        if epoch_idx % min(100, max(period, 20)) == 0:
            print(epoch_idx, avg_loss, '....', [l.item() for l in avg_loss_tuple])

        if avg_loss_tuple[-1] < 1e-1:
            if avg_loss < min_loss:
                best_model = m
                min_loss = avg_loss
                tol = tolerance
            else:
                tol -= 1

        if tol == 0:
            print(epoch_idx, avg_loss, '....', [l.item() for l in avg_loss_tuple])
            break

    if best_model is None:
        best_model = m

    with torch.no_grad():
        best_model.beta_thr = 1
        _, _, probs, _, _ = best_model.compute_loss(pred, target, X_list)
        best_model.beta_thr = 0

    return best_model, probs.mean(dim=0)


def prepare_input(inp, k, inp_flag=True, max=1, entropy=True, vote=False, norm=False):
    X = inp[k:].permute(1, 2, 0)

    X_list = []

    if inp_flag:
        X_list.append(X)

    if max:
        X_list.append(torch.sort(X, 1)[0][:, -max:])

    if entropy:
        pred_log = torch.log(X)
        conf = 1 + torch.sum(X * pred_log, dim=1) / math.log(X.shape[1])
        X_list.append(conf.unsqueeze(1))

    if vote:
        vote_ratios = np.apply_along_axis(lambda x: np.unique(x, return_counts=True)[1][-1] / (k + 1), axis=0,
                                          arr=inp.argmax(dim=2).cpu().numpy())
        X_list.append(torch.tensor(vote_ratios, device=inp.device).unsqueeze(1).unsqueeze(1).float())

    X = torch.concat(X_list, dim=1)
    X = X.reshape(X.shape[0], -1)

    X_mean = None
    X_std = None

    if norm:
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X = (X - X_mean) / X_std

    return X, (X_mean, X_std)
