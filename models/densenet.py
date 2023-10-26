'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import conv1x1, conv3x3, exit_classifier


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, self.expansion * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion * growth_rate)
        self.conv2 = nn.Conv2d(self.expansion * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block_type, nblocks, growth_rate=12, reduction=0.5, num_classes=10, ee_layer_locations=[]):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_classes = num_classes

        num_planes = 2 * growth_rate
        if num_classes == 1000:
            self.conv1 = nn.Sequential(nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False),
                                       nn.BatchNorm2d(num_planes),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.ee_classifiers = []
        self.ee_layer_locations = ee_layer_locations

        self.dense1 = self._make_dense_layers(block_type, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        if 1 in self.ee_layer_locations:
            self.ee_classifiers.append(exit_classifier(1, out_planes, num_classes=self.num_classes, reduction=block_type.expansion))

        self.dense2 = self._make_dense_layers(block_type, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        if 2 in self.ee_layer_locations:
            self.ee_classifiers.append(exit_classifier(2, out_planes, num_classes=self.num_classes, reduction=block_type.expansion))

        self.dense3 = self._make_dense_layers(block_type, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        if 3 in self.ee_layer_locations:
            self.ee_classifiers.append(exit_classifier(3, out_planes, num_classes=self.num_classes, reduction=block_type.expansion))

        self.dense4 = self._make_dense_layers(block_type, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self.ee_classifiers = nn.ModuleList(self.ee_classifiers)

    def _make_dense_layers(self, block_type, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block_type(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def _calculate_confidence(self, pred):
        pred_soft = F.softmax(pred, dim=1)
        pred_log = F.log_softmax(pred, dim=1)
        conf = 1 + torch.sum(pred_soft * pred_log, dim=1) / math.log(pred.shape[1])
        return conf

    def forward(self, x, epoch=0, manual_early_exit_index=0, conf_early_exit=True):
        out = self.conv1(x)
        preds = []

        out = self.trans1(self.dense1(out))
        if 1 in self.ee_layer_locations:
            pred = self.ee_classifiers[0](out)
            preds.append(pred)
            if manual_early_exit_index == 1:
                out = None

        if out is not None:
            out = self.trans2(self.dense2(out))
            if 2 in self.ee_layer_locations:
                pred = self.ee_classifiers[1](out)
                preds.append(pred)
                if manual_early_exit_index == 2:
                    out = None

        if out is not None:
            out = self.trans3(self.dense3(out))
            if 3 in self.ee_layer_locations:
                pred = self.ee_classifiers[2](out)
                preds.append(pred)
                if manual_early_exit_index == 3:
                    out = None

        if out is not None:
            out = self.dense4(out)
            out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            preds.append(out)

        conf_scores_list = []
        for pred in preds:
            conf_scores = self._calculate_confidence(pred)
            # conf_score_mean = conf_scores.mean()
            conf_scores_list.append(conf_scores.detach().cpu().numpy().tolist())

        return preds, conf_scores_list


def densenet121_1(args, params):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=args.num_classes)


def densenet121_4(args, params):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=args.num_classes, ee_layer_locations=[1, 2, 3])
