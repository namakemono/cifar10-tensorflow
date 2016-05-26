import numpy as np
import tensorflow as tf

def conv2d(x, W, strides=1):
    return tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')

def weight_variable(shape, wd=0.0001):
    var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def loss(labels, logits):
    return -tf.reduce_mean(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

def train(total_loss):
    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)

def conv(x, n, strides=1):
    W, b = weight_variable([3,3,channels(x),n]), bias_variable([n])
    return conv2d(x, W, strides) + b

def dense(x, n):
    W, b = weight_variable([volume(x), n]), bias_variable([n])
    return tf.matmul(x, W) + b

def activation(x):
    return tf.nn.relu(x)

def max_pool(x, ksize=2, strides=2):
    return tf.nn.max_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='SAME')

def avg_pool(x, ksize=2, strides=2):
    return tf.nn.avg_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='SAME')

def channels(x):
    return int(x.get_shape()[-1])

def volume(x):
    return np.prod([d for d in x.get_shape()[1:].as_list()])

def flatten(x):
    return tf.reshape(x, [-1, volume(x)])

def batch_normalization(x):
    eps = 1e-5
    gamma = bias_variable([channels(x)])
    beta = bias_variable([channels(x)])
    mean, variance = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps, 'bn') 

def accuracy_score(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
    return accuracy

