#!/bin/sh

# 各種ソルバーでの比較実験
# python train.py --class_name Cifar10Classifier_ResNet32_Momentum
# python train.py --class_name Cifar10Classifier_ResNet32_Adadelta
# python train.py --class_name Cifar10Classifier_ResNet32_Adagrad
# python train.py --class_name Cifar10Classifier_ResNet32_Adam
# python train.py --class_name Cifar10Classifier_ResNet32_RMSProp

# Various usages of activation
# python train.py --class_name Cifar10Classifier_ResNet32_BNAfterAddition
# python train.py --class_name Cifar10Classifier_ResNet32_ReLUBeforeAddition
python train.py --class_name Cifar10Classifier_ResNet32_ReLUOnlyPreActivation 
python train.py --class_name Cifar10Classifier_ResNet32_FullPreActivation
python train.py --class_name Cifar10Classifier_ResNet32_NoActivation

