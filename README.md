# Adaptive Deep Neural Network Inference Optimization with EENet (WACV 2024)

Well-trained deep neural networks (DNNs) treat all test samples equally during prediction. Adaptive DNN inference with early exiting leverages the observation that some test examples can be easier to predict than others. This paper presents EENet, a novel early-exiting scheduling framework for multi-exit DNN models. Instead of having every sample go through all DNN layers during prediction, EENet learns an early exit scheduler, which can intelligently terminate the inference earlier for certain predictions, which the model has high confidence of early exit. As opposed to previous early-exiting solutions with heuristics-based methods, our EENet framework optimizes an early-exiting policy to maximize model accuracy while satisfying the given per-sample average inference budget. Extensive experiments are conducted on four computer vision datasets (CIFAR-10, CIFAR-100, ImageNet, Cityscapes) and two NLP datasets (SST-2, AgNews). The results demonstrate that the adaptive inference by EENet can outperform the representative existing early exit techniques. We also perform a detailed visualization analysis of the comparison results to interpret the benefits of EENet.

## Citation

If you find this repository useful, please cite our paper. Thank you!
```
@inproceedings{ilhan2023eenet,
  title={Adaptive Deep Neural Network Inference Optimization with EENet},
  author={Ilhan, Fatih and Chow, Ka-Ho and Huang, Tiansheng and Hu, Sihao and Tekin, Selim and Wei, Wenqi and Wu, Yanzhao and Lee, Myungjin and Kompella, Ramana and Latapie, Hugo and Liu, Gaowen and Liu, Ling},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

## Setup
Python 3.8
Pytorch 1.12
Transformers 4.11

## Usage

CIFAR, AgNews and SST-2 datasets are automatically downloaded. You can download the pre-trained models from [this link](https://drive.google.com/file/d/12UZ34_z3h3ecBQlaJzTIugAoMNeAC9tH/view?usp=sharing).

seconds.csv in each output folder contains latencies up to each exit for that model, which would depend on the environment and may require manual modification based on measurements.

### Train a multi-exit model:

``python main.py --data-root <DATA_PATH> --data <DATASET> --arch <MODEL_ARCH> --use-valid``

For instance, DenseNet121 with four exits on CIFAR100:

``python main.py --data-root datasets --data cifar100 --arch densenet121_4 --use-valid``

### Budgeted adaptive inference with a multi-exit model:

``python main.py --data-root <DATA_PATH>--data <DATASET> --save-path <SAVE_PATH> --arch <MODEL_ARCH> --evalmode dynamic --use-valid --evaluate-from <MODEL_CHKP> --val_budget <VAL_BUDGET>  --conf_mode  <CONF_MODE> --edm <EXIT_DIST_METHOD>``

For instance, BERT with four exits on AgNews with the budget of 125 ms/sample using EENet:

``python main.py --data-root datasets --data ag_news --save-path outputs/bert_4_None_ag_news --arch bert_4 --evalmode dynamic --use-valid --evaluate-from outputs/bert_4_None_ag_news/save_models/bert_agnews.tar --val_budget 125  --conf_mode nn --edm nn``

using BranchyNet (entropy):

``python main.py --data-root datasets --data ag_news --save-path outputs/bert_4_None_ag_news --arch bert_4 --evalmode dynamic --use-valid --evaluate-from outputs/bert_4_None_ag_news/save_models/bert_agnews.tar --val_budget 125 --conf_mode entropy --edm exp``

For instance, DenseNet with four exits on CIFAR100 with the budget of 6.5 ms/sample using EENet:

``python main.py --data-root datasets --data cifar100 --save-path outputs/densenet121_4_None_cifar100 --arch densenet121_4 --evalmode dynamic --use-valid --evaluate-from outputs/densenet121_4_None_cifar100/save_models/cifar100_densenet.tar --val_budget 6.5 --conf_mode nn --edm nn``

using MSDNet (max score):

``python main.py --data-root datasets --data cifar100 --save-path outputs/densenet121_4_None_cifar100 --arch densenet121_4 --evalmode dynamic --use-valid --evaluate-from outputs/densenet121_4_None_cifar100/save_models/cifar100_densenet.tar --val_budget 6.5 --conf_mode maxpred --edm exp``

For instance, ResNet with three exits on CIFAR10 with the budget of 2.5 ms/sample using EENet:

``python main.py --data-root datasets --data cifar10 --save-path outputs/resnet56_3_None_cifar10 --arch resnet56_3 --evalmode dynamic --use-valid --evaluate-from outputs/resnet56_3_None_cifar10/save_models/cifar10_resnet.tar --val_budget 2.5 --conf_mode nn --edm nn``

using MSDNet (max score):

``python main.py --data-root datasets --data cifar10 --save-path outputs/resnet56_3_None_cifar10 --arch resnet56_3 --evalmode dynamic --use-valid --evaluate-from outputs/resnet56_3_None_cifar10/save_models/cifar10_resnet.tar --val_budget 2.5 --conf_mode maxpred --edm exp``


### Parameters

All training/inference/model parameters are controlled from ``config.py``.
