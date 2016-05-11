import cv2
import os
import urllib
import cPickle
import numpy as np

def unpickle(filename):
    with open(filename, 'rb') as fp:
        return cPickle.load(fp)

def shuffle(images, labels):
    perm = np.arange(len(labels))
    np.random.shuffle(perm)
    return np.asarray(images)[perm], np.asarray(labels)[perm]

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    return image[offset_width:offset_width+target_width, offset_height:offset_height+target_height]

def random_contrast(image, lower, upper, seed=None):
    contrast_factor = np.random.uniform(lower, upper)
    avg = np.mean(image)
    return (image - avg) * contrast_factor + avg

def random_brightness(image, max_delta, seed=None):
    delta = np.random.randint(-max_delta, max_delta)
    return image - delta

def random_blur(image, size):
    if np.random.random() < 0.5:
        image = cv2.blur(image, size)
    return image

def normalize(image):
    return image / 255.0

def per_image_whitening(image):
    return (image - np.mean(image)) / np.std(image)

def random_flip_left_right(image):
    if np.random.random() < 0.5:
        image = cv2.flip(image, 1)
    return image

def random_flip_up_down(image):
    if np.random.random() < 0.5:
        image = cv2.flip(image, 0)
    return image  

def random_crop(image, size):
    if len(image.shape):
        W, H, D = image.shape
        w, h, d = size
    else:
        W, H = image.shape
        w, h = size
    left, top = np.random.randint(W - w + 1), np.random.randint(H - h + 1)
    return image[left:left+w, top:top+h]

