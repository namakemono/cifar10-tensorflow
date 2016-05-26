import tensorflow as tf
import cv2
import os
import urllib
import numpy as np
import image_processing
from os import path
ROOT = path.dirname(path.dirname(path.abspath(__file__)))

def distort(image, is_train=True):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(float)
    if is_train:
        # I = np.zeros((40, 40, 3))
        # I[4:36, 4:36] = image
        # image = image_processing.random_crop(I, (32, 32, 3))
        image = image_processing.random_crop(image, (24, 24, 3))
        image = image_processing.random_flip_left_right(image)
        # image = image_processing.random_brightness(image, max_delta=63)
        # image = image_processing.random_contrast(image, lower=0.2, upper=1.8)
    else:
        pass
        image = image_processing.crop_to_bounding_box(image, 4, 4, 24, 24)
    image = image_processing.per_image_whitening(image)
    return image

def load_cifar10(is_train=True):
    if is_train:
        filenames = [ROOT + "/input/cifar-10-batches-py/data_batch_%d" % j for j in range(1, 6)]
    else:
        filenames = [ROOT + "/input/cifar-10-batches-py/test_batch"]
    images, labels = [], []
    for filename in filenames:
        cifar10 = image_processing.unpickle(filename)
        for i in range(len(cifar10["labels"])):
            images.append(distort(cifar10["data"][i], is_train))
        labels += cifar10["labels"]
    return image_processing.shuffle(images, image_processing.dense_to_one_hot(np.asarray(labels)))

