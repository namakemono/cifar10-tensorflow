# CIFAR-10 via TensorFlow

## Install
```
sh install.sh
```

## Usage
network.pyにあるCifar10Classifier_XXXをtrain.pyの下の方に突っ込んで以下のコマンドを実行する．
```
# Residual Network(32 layers)を訓練させたい場合
python train.py --class_name Cifar10Classifier_ResNet32
```
output に数値計算結果が出力され，modelsにモデルが生成されます．


## Performance

| Name                    | Precision       | Memo                      |
|-------------------------|-----------------|---------------------------|
|Cifar10Classifier_01     | 83.11%          |                           |
|Cifar10Classifier_02     | 87.00%          |                           |
|Cifar10Classifier_03     | 87.25%          |                           |
|Cifar10Classifier_04     | 87.67%          |                           |
|Cifar10Classifier_05     | 87.17%          |                           |
|Cifar10Classifier_06     | 86.74%          |                           |
|Cifar10Classifier_ResNet20     | 91.07%          | [2]                 |
|Cifar10Classifier_ResNet32     | 92.04%          | [2]                 |
|Cifar10Classifier_ResNet44     | 91.93%          | [2]                 |
|Cifar10Classifier_ResNet56     | 92.38%          | [2]                 |
|Cifar10Classifier_ResNet110     | 92.94%          | [2]                 |

## Environment

| Name     | Description           |
|----------|-----------------------|
|GPU       | GeForce GTX TITAN X   |
|OS        | Ubuntu 16.04 LTS      |
|Library   | TensorFlow 0.8.0      |

## ResNet

### 層数でのテストデータのテスト誤差(%)

![ResNet on CIFAR-10](figures/resnet.layers.png)

1epochは訓練データ5万枚を一周学習させた回数

### 各種ソルバーでのテスト誤差(%)

![各種ソルバーでのResNet32のテスト誤差](figures/resnet.solvers.png)

| Name                    | Test Error    |
|-------------------------|---------------|
| Original Paper          | 8.27%         |
| Adadelta(LR 1e-3)       | 31.03%        |
| Adagrad(LR 1e-2)        | 15.90%        |
| RMSProp(LR 1e-3)        | 10.97%        |

注). LRはLearning Rateの意

### Batch NormとReLUの位置でのテスト誤差

![Batch NormとReLUの位置違いでのテスト誤差](figures/resnet.mapping.png)

| Name                     | Test Error |
|--------------------------|------------|
| Original Paper           | 8.27%      |
| BN after addition        | 8.89%      |
| ReLU before addition     | 9.54%      |
| ReLU only pre activation | 8.82%      |
| ful pre-activation       | 10.03%     |
| No ReLU                  | 8.85%      |

## References
- [1]. Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).

Batch Normの仕組みについて記載

- [2]. He, Kaiming, et al. "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385 (2015).

ImageNet 2015優勝アルゴリズム.

- [3]. He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE International Conference on Computer Vision. 2015.

ResNetなど多層のネットワークを構築する上で必要な重みの初期化方法が載っている．

- [4]. He, Kaiming, et al. "Identity mappings in deep residual networks." arXiv preprint arXiv:1603.05027 (2016).

Residual Networkの解析が行われている．

- [5]. Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).

ResNet構築に必要なGlobal Average Poolingについて記載されている．

- [?]. [Survey - Deep Residual Learning for Image Recognition](http://qiita.com/supersaiakujin/items/935bbc9610d0f87607e8), 2016/03/01

[2].のサーベイ

- [?]. [Survey - Identity Mappings in Deep Residual Networks](http://qiita.com/supersaiakujin/items/eaa0fe0460c470c28bd8), 2016/03/30

[4].のサーベイ

