import argparse
import datetime
import os


def modify_args(args):
    if args.use_gpu and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    elif args.data == 'tinyimagenet':
        args.num_classes = 200
    elif args.data == 'imagenet':
        args.num_classes = 1000
    elif args.data == 'sst2':
        args.num_classes = 2
    elif args.data == 'ag_news':
        args.num_classes = 4

    if not hasattr(args, "save_path") or args.save_path is None:
        args.save_path = f"outputs/{args.arch}_{args.evalmode}_{args.data}_{format(str(datetime.datetime.now()))}"

    if args.data.startswith('cifar'):
        args.image_size = (32, 32)
    elif args.data == 'tinyimagenet':
        args.image_size = (64, 64)
    elif args.data == 'imagenet':
        args.image_size = (224, 224)
    elif args.data == 'sst2':
        args.image_size = (1, 512)
    elif args.data == 'ag_news':
        args.image_size = (1, 512)
    return args


model_names = ['msdnet35_5', 'resnet56_3', 'densenet121_4', 'bert_4']

arg_parser = argparse.ArgumentParser(
    description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save-path', default=None,
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default=None,
                       choices=['anytime', 'dynamic'],
                       help='which mode to evaluate')
exp_group.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')
exp_group.add_argument('--use_gpu', default=False, type=bool, help='Use CPU if False')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'imagenet', 'sst2', 'ag_news'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet56_1',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet56_1)')

# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size')

# inference related
optim_group = arg_parser.add_argument_group('inference', 'inference setting')
optim_group.add_argument('-edm', '--exit-distribution-method', type=str, default='nn',
                         choices=['exp', 'nn'],
                         help='exit distribution method selection (exp: msdnet and branchynet, nn: ours)')
optim_group.add_argument('--conf_mode', type=str, default='nn',
                         choices=['maxpred', 'entropy', 'nn'],
                         help='confusion measure selection (entropy: branchynet, maxpred: msdnet, nn: ours)')
optim_group.add_argument('--inference-save-filename', type=str, default='dynamic',
                         help='name of the file to save inference results')
optim_group.add_argument('--val_budget', type=float,
                         help='average inference budget per sample, scans range if not provided')
