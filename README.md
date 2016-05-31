# CIFAR-10 via TensorFlow

## Install
```
sh install.sh
```

## Usage
network.pyにあるCifar10Classifier_XXXをtrain.pyの下の方に突っ込んで以下のコマンドを実行する．
```
python train.py
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

![ResNet on CIFAR-10](figures/resnet_cifar10.png)

1epochは訓練データ5万枚を一周学習させた回数

## References
- [1]. Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).


- [2]. He, Kaiming, et al. "Deep Residual Learning for Image Recognition." arXiv preprint arXiv:1512.03385 (2015).



- [3]. He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE International Conference on Computer Vision. 2015.

重みの初期化方法が載っている．
