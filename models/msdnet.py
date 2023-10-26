import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import pdb
import random


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck, bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down', bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal', bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal', bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args, params):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * params['growth_factor'][0], kernel=3, stride=1, padding=1))
        elif args.data == 'tinyimagenet':
            self.layers.append(ConvBasic(nIn, nOut * params['growth_factor'][0], kernel=3, stride=2, padding=1))
        elif args.data == 'imagenet':
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * params['growth_factor'][0], 7, 2, 3),
                nn.BatchNorm2d(nOut * params['growth_factor'][0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * params['growth_factor'][0]

        for i in range(1, params['num_scales']):
            self.layers.append(ConvBasic(nIn, nOut * params['growth_factor'][i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * params['growth_factor'][i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, params, in_scales=None, out_scales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.in_scales = in_scales if in_scales is not None else params['num_scales']
        self.out_scales = out_scales if out_scales is not None else params['num_scales']

        self.num_scales = params['num_scales']
        self.discard = self.in_scales - self.out_scales

        self.offset = self.num_scales - self.out_scales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * params['growth_factor'][self.offset - 1]
            nIn2 = nIn * params['growth_factor'][self.offset]
            _nOut = nOut * params['growth_factor'][self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, params['bottleneck'],
                                              params['bn_factor'][self.offset - 1],
                                              params['bn_factor'][self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * params['growth_factor'][self.offset],
                                          nOut * params['growth_factor'][self.offset],
                                          params['bottleneck'],
                                          params['bn_factor'][self.offset]))

        for i in range(self.offset + 1, self.num_scales):
            nIn1 = nIn * params['growth_factor'][i - 1]
            nIn2 = nIn * params['growth_factor'][i]
            _nOut = nOut * params['growth_factor'][i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, params['bottleneck'],
                                              params['bn_factor'][i - 1],
                                              params['bn_factor'][i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.out_scales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.out_scales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.out_scales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class SwitcherModule(nn.Module):
    def __init__(self, channel, num_switches):
        super(SwitcherModule, self).__init__()
        self.num_switches = num_switches
        self.conv = ConvBasic(3, channel)
        self.linear = nn.Linear(channel*192, num_switches)

    def forward(self, x):
        x = x.view(self.conv(x).size(0), -1)
        x = self.linear(x)
        return F.softmax(x)


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        return self.linear(res)


class MSDNet(nn.Module):
    def __init__(self, args, params):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.num_blocks = params['num_blocks']
        self.steps = [params['base']]
        self.args = args

        self.switcher = SwitcherModule(16, self.num_blocks)

        # self.exit_prob_list = [1/(2**i) for i in range(1, self.num_blocks)] + [1/(2**(self.num_blocks-1))]
        # self.exit_prob_list = [1 / self.num_blocks] * self.num_blocks
        self.exit_prob_list = []

        self.exit_loss_weights = 1 / np.array(self.exit_prob_list)[::-1].cumsum()[::-1]
        self.exit_loss_weights /= self.exit_loss_weights.sum()

        self.conf_thresholds = [0.95 - i * 0.01 for i in range(self.num_blocks)]

        n_layers_all, n_layer_curr = params['base'], 0
        for i in range(1, self.num_blocks):
            self.steps.append(params['step'] if params['step_mode'] == 'even' else params['step'] * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = params['num_channels']
        for i in range(self.num_blocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, params, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * params['growth_factor'][-1], 100))
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * params['growth_factor'][-1], 10))
            elif args.data == 'tinyimagenet':
                self.classifier.append(
                    self._build_classifier_cifar(nIn * params['growth_factor'][-1], 200))
            elif args.data == 'imagenet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * params['growth_factor'][-1], 1000))
            else:
                raise NotImplementedError

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, params, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args, params)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            if params['prune'] == 'min':
                in_scales = min(params['num_scales'], n_layer_all - n_layer_curr + 2)
                out_scales = min(params['num_scales'], n_layer_all - n_layer_curr + 1)
            elif params['prune'] == 'max':
                interval = math.ceil(1.0 * n_layer_all / params['num_scales'])
                in_scales = params['num_scales'] - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                out_scales = params['num_scales'] - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            growth_factor = params['growth_factor']
            growth_rate = params['growth_rate']
            layers.append(MSDNLayer(nIn, growth_rate, params, in_scales, out_scales))
            print('|\t\tin_scales {} out_scales {} inChannels {} outChannels {}\t\t|'.format(in_scales, out_scales, nIn, growth_rate))

            nIn += growth_rate
            if params['prune'] == 'max' and in_scales > out_scales and params['reduction'] > 0:
                offset = params['num_scales'] - out_scales
                layers.append(self._build_transition(nIn, math.floor(1.0 * params['reduction'] * nIn), out_scales, offset, growth_factor))
                _t = nIn
                nIn = math.floor(1.0 * params['reduction'] * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(
                    1.0 * params['reduction'] * _t)))
            elif params['prune'] == 'min' and params['reduction'] > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = params['num_scales'] - out_scales
                layers.append(self._build_transition(nIn, math.floor(1.0 * params['reduction'] * nIn), out_scales, offset, growth_factor))

                nIn = math.floor(1.0 * params['reduction'] * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, out_scales, offset, growth_factor):
        net = []
        for i in range(out_scales):
            net.append(ConvBasic(nIn * growth_factor[offset + i],
                                 nOut * growth_factor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def _calculate_confidence(self, pred):
        pred_soft = F.softmax(pred, dim=1)
        pred_log = F.log_softmax(pred, dim=1)
        conf = 1 + torch.sum(pred_soft * pred_log, dim=1) / math.log(pred.shape[1])
        return conf

    def forward(self, x, epoch=0, manual_early_exit_index=0, conf_early_exit=True):

        res = []
        conf_scores_list = []

        # if self.exit_prob_list and epoch >= 0:
        #     num_blocks = random.choices(list(range(self.num_blocks)), self.exit_prob_list, k=1)[0] + 1
        # else:
        #     num_blocks = self.num_blocks

        num_iter = self.num_blocks
        if manual_early_exit_index != 0:
            num_iter = min(manual_early_exit_index, self.num_blocks)

        for i in range(num_iter):
            x = self.blocks[i](x)

            pred = self.classifier[i](x)
            conf_scores = self._calculate_confidence(pred)
            conf_score_mean = conf_scores.mean()
            conf_scores_list.append(conf_scores.detach().cpu().numpy().tolist())

            res.append(pred)

        return res, conf_scores_list


def msdnet35_5(args, params):
    return MSDNet(args, params)