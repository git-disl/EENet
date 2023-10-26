import os

import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


def get_dataloaders(args, batch_size):
    train_set, val_set, test_set = None, None, None

    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                     ]))
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize
                                      ]))
        test_set = datasets.CIFAR100(args.data_root, train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         normalize
                                     ]))
    elif args.data == 'tinyimagenet':
        # tinyimagenet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2302, 0.2265, 0.2262])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ]))
        test_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    elif args.data == 'imagenet':
        # imagenet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        test_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    elif args.data == 'sst2':
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

        task = "sst2"
        model_checkpoint = "bert-base-uncased"
        dataset = load_dataset("glue", task)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        sentence1_key, sentence2_key = task_to_keys[task]

        def preprocess_function(examples):
            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

        sentence1_key, sentence2_key = task_to_keys[task]

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_set = encoded_dataset['train']
        val_set = encoded_dataset['validation']
        test_set = encoded_dataset['test']
        test_df = pd.read_csv('datasets/sst2/test.tsv', header=None, sep='\t')
        synch_list = [test_set['sentence'].index(s.lower().replace('-lrb-', '(').replace('-rrb-', ')'))
                      for s in test_df.iloc[:, 0].tolist()]
        synch_list = [synch_list.index(i) for i in range(len(synch_list))]
        new_labels = [test_df.iloc[:, -1].tolist()[x] for x in synch_list]

        def change_label(data, idx):
            data['label'] = new_labels[idx]
            return data

        test_set = test_set.map(change_label, with_indices=True)
    elif args.data == 'ag_news':
        model_checkpoint = "bert-base-uncased"
        dataset = load_dataset(args.data)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        sentence1_key, sentence2_key = 'text', None

        def preprocess_function(examples):
            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_set = encoded_dataset['train']
        test_set = encoded_dataset['test']

    train_loader, val_loader, test_loader = None, None, None

    def collate_fn(data):
        return (pad_sequence([torch.tensor(d['input_ids']) for d in data], batch_first=True, padding_value=0),
                torch.tensor([d['label'] for d in data]))

    if 'bert' in args.arch:
        cfn = collate_fn
    else:
        cfn = None

    if args.use_valid:
        if val_set is None:
            train_set_index = torch.randperm(len(train_set))
            if os.path.exists(os.path.join(args.save_path, 'index.pth')):
                print('!!!!!! Load train_set_index !!!!!!')
                train_set_index = torch.load(os.path.join(args.save_path, 'index.pth'))
            else:
                print('!!!!!! Save train_set_index !!!!!!')
                torch.save(train_set_index, os.path.join(args.save_path, 'index.pth'))
            if args.data.startswith('cifar'):
                num_sample_valid = 5000
            elif args.data == 'tinyimagenet':
                num_sample_valid = 10000
            elif args.data == 'imagenet':
                num_sample_valid = 50000
            elif args.data == 'sst2':
                num_sample_valid = 872
            elif args.data == 'ag_news':
                num_sample_valid = 10000
            else:
                raise NotImplementedError

            train_indices = train_set_index[:-num_sample_valid].tolist()
            val_indices = train_set_index[-num_sample_valid:].tolist()
            val_set = train_set
        else:
            train_indices = torch.arange(len(train_set)).tolist()
            val_indices = torch.arange(len(val_set)).tolist()

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_indices),
                num_workers=args.workers, pin_memory=True, collate_fn=cfn)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, shuffle=False,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    val_indices),
                num_workers=args.workers, pin_memory=True, collate_fn=cfn)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, collate_fn=cfn)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, collate_fn=cfn)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, collate_fn=cfn)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
