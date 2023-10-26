class Config:
    def __init__(self):
        self.training_params = {
            'cifar10': {
                'resnet56_1': {'batch_size': 128,
                               'num_epoch': 150,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [50, 100],
                               'weight_decay': 5e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd',
                               },
                'resnet56_3': {'batch_size': 128,
                               'num_epoch': 150,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [50, 100],
                               'weight_decay': 5e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd',
                               }
            },
            'cifar100': {
                'densenet121_1': {'batch_size': 128,
                                  'num_epoch': 150,
                                  'lr': 0.1,
                                  'lr_type': 'multistep',
                                  'decay_rate': 0.1,
                                  'decay_epochs': [50, 100],
                                  'weight_decay': 5e-4,
                                  'momentum': 0.9,
                                  'optimizer': 'sgd',
                                  },
                'densenet121_4': {'batch_size': 128,
                                  'num_epoch': 150,
                                  'lr': 0.1,
                                  'lr_type': 'multistep',
                                  'decay_rate': 0.1,
                                  'decay_epochs': [50, 100],
                                  'weight_decay': 5e-4,
                                  'momentum': 0.9,
                                  'optimizer': 'sgd',
                                  }
            },
            'imagenet': {
                'msdnet35_5': {'batch_size': 64,
                               'num_epoch': 200,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [75, 150],
                               'weight_decay': 1e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd',
                               }
            },
            'sst2': {
                'bert_1': {
                    'batch_size': 16,
                    'num_epoch': 20,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd',

                },
                'bert_4': {
                    'batch_size': 16,
                    'num_epoch': 20,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd',

                }
            },
            'ag_news': {
                'bert_1': {
                    'batch_size': 16,
                    'num_epoch': 20,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd',

                },
                'bert_4': {
                    'batch_size': 16,
                    'num_epoch': 20,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd',

                }
            }
        }
        self.model_params = {
            'cifar10': {
                'resnet56_1': {'ee_layer_locations': [],
                               'ee_num_conv_layers': [],
                               'num_blocks': 1},
                'resnet56_3': {'ee_layer_locations': [9, 18],
                               'ee_num_conv_layers': [3, 3],
                               'num_blocks': 3}
            },
            'cifar100': {
                'densenet121_1': {'ee_layer_locations': [],
                                  'ee_num_conv_layers': [],
                                  'num_blocks': 1},
                'densenet121_4': {'ee_layer_locations': [1, 2, 3],
                                  'ee_num_conv_layers': [3, 3, 3],
                                  'num_blocks': 4}
            },
            'imagenet': {
                'msdnet35_5': {'base': 7,
                               'step': 7,
                               'num_scales': 4,
                               'step_mode': 'even',
                               'num_channels': 32,
                               'growth_rate': 16,
                               'growth_factor': [1, 2, 4, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 5,
                               'reduction': 0.5}
            },
            'sst2': {
                'bert_1': {'ee_layer_locations': [],
                           'num_blocks': 1},
                'bert_4': {'ee_layer_locations': [3, 6, 9],
                           'num_blocks': 4},
            },
            'ag_news': {
                'bert_1': {'ee_layer_locations': [],
                           'num_blocks': 1},
                'bert_4': {'ee_layer_locations': [3, 6, 9],
                           'num_blocks': 4},
            }
        }
        self.inference_params = {
            'cifar10': {
                'resnet56_3': {'weight_decay': 1e-2,
                               'beta_ce': 1,
                               'alpha_ce': 1e-3,
                               'alpha_cost': 1e-2,
                               'num_epoch': 1000,
                               'bs': 512,
                               'lr': 3e-5,
                               'hidden_dim_rate': 0.5,
                               'period': 10}
            },
            'cifar100': {
                'densenet121_4': {'weight_decay': 1e-2,
                                  'beta_ce': 1,
                                  'alpha_ce': 1e-3,
                                  'alpha_cost': 1e-2,
                                  'num_epoch': 1000,
                                  'bs': 512,
                                  'lr': 3e-5,
                                  'hidden_dim_rate': 0.5,
                                  'period': 10}
            },
            'imagenet': {
                'msdnet35_5': {'alpha_ce': 1e-3,
                               'alpha_cost': 1e-2,
                               'num_epoch': 1000,
                               'bs': 512,
                               'lr': 3e-5,
                               'hidden_dim_rate': 0.5}
            },
            'sst2': {
                'bert_4': {'weight_decay': 1e-2,
                           'beta_ce': 1,
                           'alpha_ce': 1e-3,
                           'alpha_cost': 1e-2,
                           'num_epoch': 1000,
                           'bs': 512,
                           'lr': 3e-5,
                           'hidden_dim_rate': 2,
                           'period': 10}
            },
            'ag_news': {
                'bert_4': {'weight_decay': 1e-2,
                           'beta_ce': 1,
                           'alpha_ce': 1e-3,
                           'alpha_cost': 1e-2,
                           'num_epoch': 1000,
                           'bs': 512,
                           'lr': 3e-5,
                           'hidden_dim_rate': 2,
                           'period': 10}
            }
        }
