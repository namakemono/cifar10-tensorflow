#!/bin/sh

# 層数でのテスト誤差比較
# python train.py --class_name Cifar10Classifier_ResNet20
# python train.py --class_name Cifar10Classifier_ResNet32
# python train.py --class_name Cifar10Classifier_ResNet44
# python train.py --class_name Cifar10Classifier_ResNet56
# python train.py --class_name Cifar10Classifier_ResNet110

# 各種ソルバーでのテスト誤差比較
# python train.py --class_name Cifar10Classifier_ResNet32_Momentum
# python train.py --class_name Cifar10Classifier_ResNet32_Adadelta
# python train.py --class_name Cifar10Classifier_ResNet32_Adagrad
# python train.py --class_name Cifar10Classifier_ResNet32_Adam
# python train.py --class_name Cifar10Classifier_ResNet32_RMSProp

# Batch NormとReLUの適用箇所でのテスト誤差比較
# python train.py --class_name Cifar10Classifier_ResNet32_BNAfterAddition
# python train.py --class_name Cifar10Classifier_ResNet32_ReLUBeforeAddition
# python train.py --class_name Cifar10Classifier_ResNet32_ReLUOnlyPreActivation 
# python train.py --class_name Cifar10Classifier_ResNet32_FullPreActivation
# python train.py --class_name Cifar10Classifier_ResNet32_NoActivation

