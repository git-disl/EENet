import torch.nn as nn
from torch.nn import Sequential


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def exit_classifier(id: int, in_planes: int, num_classes: int, num_conv_layer: int = 3, reduction: int=1):
    if reduction == 1:
        conv_list = [conv3x3(in_planes, in_planes) for _ in range(num_conv_layer)]
    else:
        conv_list = [conv3x3(in_planes, int(in_planes/reduction))]
        in_planes = int(in_planes/reduction)
        conv_list.extend([conv3x3(in_planes, in_planes) for _ in range(num_conv_layer-1)])
    bn_list = [nn.BatchNorm2d(in_planes) for _ in range(num_conv_layer)]
    relu_list = [nn.ReLU() for _ in range(num_conv_layer)]
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    flatten = nn.Flatten()
    fc = nn.Linear(in_planes, num_classes)

    layers = []
    for i in range(num_conv_layer):
        layers.append(conv_list[i])
        layers.append(bn_list[i])
        layers.append(relu_list[i])
    layers.append(avg_pool)
    layers.append(flatten)
    layers.append(fc)

    return Sequential(*layers)
    # return Sequential(OrderedDict([(f'ee_conv_{id}_{i}', c) for i, c in enumerate(conv_list)] + [(f'ee_pool_{id}', avg_pool), (f'ee_flat_{id}', flatten), (f'ee_fc_{id}', fc)]))

