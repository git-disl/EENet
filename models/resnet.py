from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from models.model_utils import conv1x1, conv3x3, exit_classifier


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, ee_layer_locations=[]):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_blocks = len(ee_layer_locations) + 1
        self.num_classes = num_classes

        ee_block_list = []
        ee_layer_list = []

        for ee_layer_idx in ee_layer_locations:
            b, l = self.find_ee_block_and_layer(layers, ee_layer_idx)
            ee_block_list.append(b)
            ee_layer_list.append(l)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1, self.ee1 = self._make_layer(block, 16, layers[0], stride=1, ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 0])
        self.layer2, self.ee2 = self._make_layer(block, 32, layers[1], stride=2, ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 1])
        self.layer3, self.ee3 = self._make_layer(block, 64, layers[2], stride=2, ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 2])
        # self.layer4, self.ee4 = self._make_layer(block, 512, layers[3], stride=2, early_exit_layers=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 3])
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block_type, planes, num_block, stride, ee_layer_locations):
        strides = [stride] + [1] * (num_block - 1)

        ee_layer_locations_ = ee_layer_locations + [num_block]
        layers = [[] for _ in range(len(ee_layer_locations_))]

        ee_classifiers = []

        if len(ee_layer_locations_) > 1:
            start_layer = 0
            counter = 0
            for i, ee_layer_idx in enumerate(ee_layer_locations_):
                for _ in range(start_layer, ee_layer_idx):
                    layers[i].append(block_type(self.in_planes, planes, strides[counter]))
                    self.in_planes = planes * block_type.expansion
                    counter += 1
                start_layer = ee_layer_idx
                if i < len(ee_layer_locations_) - 1:
                    if ee_layer_idx == 0:
                        ee_classifiers.append(exit_classifier(i, self.in_planes, num_classes=self.num_classes, reduction=block_type.expansion))
                    else:
                        ee_classifiers.append(exit_classifier(i, planes * block_type.expansion, num_classes=self.num_classes, reduction=block_type.expansion))
        else:
            for i in range(num_block):
                layers[0].append(block_type(self.in_planes, planes, strides[i]))
                self.in_planes = planes * block_type.expansion

        return nn.ModuleList([nn.Sequential(*l) for l in layers]), nn.ModuleList(ee_classifiers)

    @staticmethod
    def find_ee_block_and_layer(layers, layer_idx):
        temp_array = np.zeros((sum(layers)), dtype=int)
        cum_array = np.cumsum(layers)
        for i in range(1, len(cum_array)):
            temp_array[cum_array[i-1]:] += 1
        block = temp_array[layer_idx]
        if block == 0:
            layer = layer_idx
        else:
            layer = layer_idx - cum_array[block-1]
        return block, layer

    def forward(self, x, epoch=0, manual_early_exit_index=0, conf_early_exit=True):
        out = F.relu(self.bn1(self.conv1(x)))
        ee_outs = []

        if manual_early_exit_index > len(self.ee1):
            manual_early_exit_index_ = 0
        else:
            manual_early_exit_index_ = manual_early_exit_index
        final_out, outs = self._block_forward(self.layer1, self.ee1, out, manual_early_exit_index_)
        if outs:
            ee_outs.extend(outs)

        if final_out is not None:
            if manual_early_exit_index > len(self.ee1) + len(self.ee2):
                manual_early_exit_index_ = 0
            elif manual_early_exit_index:
                manual_early_exit_index_ = manual_early_exit_index - len(self.ee1)
            else:
                manual_early_exit_index_ = manual_early_exit_index
            final_out, outs = self._block_forward(self.layer2, self.ee2, final_out, manual_early_exit_index_)
            if outs:
                ee_outs.extend(outs)

        if final_out is not None:
            if manual_early_exit_index > len(self.ee1) + len(self.ee2) + len(self.ee3):
                manual_early_exit_index_ = 0
            elif manual_early_exit_index:
                manual_early_exit_index_ = manual_early_exit_index - len(self.ee1) - len(self.ee2)
            else:
                manual_early_exit_index_ = manual_early_exit_index
            final_out, outs = self._block_forward(self.layer3, self.ee3, final_out, manual_early_exit_index_)
            if outs:
                ee_outs.extend(outs)

        preds = ee_outs

        if final_out is not None:
            out = F.adaptive_avg_pool2d(final_out, (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            preds.append(out)

        conf_scores_list = []

        for pred in preds:
            conf_scores = self._calculate_confidence(pred)
            # conf_score_mean = conf_scores.mean()
            conf_scores_list.append(conf_scores.detach().cpu().numpy().tolist())

        return preds, conf_scores_list

    def _calculate_confidence(self, pred):
        pred_soft = F.softmax(pred, dim=1)
        pred_log = F.log_softmax(pred, dim=1)
        conf = 1 + torch.sum(pred_soft * pred_log, dim=1) / math.log(pred.shape[1])
        return conf

    def _block_forward(self, layers, ee_classifiers, x, early_exit=0):
        outs = []
        for i in range(len(layers)-1):
            x = layers[i](x)
            outs.append(ee_classifiers[i](x))
            if early_exit == i + 1:
                break
        if early_exit == 0:
            final_out = layers[-1](x)
        else:
            final_out = None
        return final_out, outs


def resnet56_1(args, params):
    return ResNet(Bottleneck, [9, 9, 9], num_classes=args.num_classes)


def resnet56_3(args, params):
    ee_layer_locations = params['ee_layer_locations']
    return ResNet(Bottleneck, [9, 9, 9], num_classes=args.num_classes, ee_layer_locations=ee_layer_locations)
