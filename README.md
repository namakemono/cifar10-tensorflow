# CIFAR-10 via TensorFlow

## Install
```
sh install.sh
```

note: perform `pip install pandas nvidia-ml-py scikit-learn` to install required libraries

## Usage
network.pyにあるCifar10Classifier_XXXをtrain.pyの下の方に突っ込んで以下のコマンドを実行する．
```
python train.py
```
output に数値計算結果が出力される．


## Performance

| Name                    | Precision       | Memo                      |
|-------------------------|-----------------|---------------------------|
|Cifar10Classifier_01     | 83.11%          |                           |
|Cifar10Classifier_02     | 87.00%          |                           |
|Cifar10Classifier_03     | 87.25%          |                           |
|Cifar10Classifier_04     | 87.67%          |                           |
|Cifar10Classifier_05     | 87.17%          |                           |
|Cifar10Classifier_06     | 86.74%          |                           |
|Cifar10Classifier_ResNet20     | 90.59%          | [2]                 |
|Cifar10Classifier_ResNet32     | 91.79%          | [2]                 |
|Cifar10Classifier_ResNet44     | 91.93%          | [2]                 |
|Cifar10Classifier_ResNet56     | 92.38%          | [2]                 |
|Cifar10Classifier_ResNet56     | 92.94%          | [2]                 |

## Environment

| Name     | Description           |
|----------|-----------------------|
|GPU       | GeForce GTX TITAN X   |
|OS        | Ubuntu 16.04 LTS      |
|Library   | TensorFlow 0.8.0      |

## ResNet

![ResNet on CIFAR-10](figures/resnet_cifar10.png)

1epochは訓練データ5万枚を一周学習させた回数

## References
- [1]. Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
- [2]. He, Kaiming, et al. "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385 (2015).

