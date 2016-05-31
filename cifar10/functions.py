import numpy as np
import tensorflow as tf

def conv2d(x, W, strides=1):
    return tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')

def weight_variable(shape, wd=1e-4):
    # c.f. 2.2 of [1]
    # [1]. He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE International Conference on Computer Vision. 2015.
    k, c = 3, shape[-2]
    var = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / (k*k*c))))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)
    print var.get_shape()
    return var

def bias_variable(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    print "bias", b.get_shape()
    return b

def loss(labels, logits):
    return -tf.reduce_mean(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

def train(total_loss):
    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)

def conv(x, n, strides=1, bias_term=True):
    W = weight_variable([3,3,channels(x),n])
    res = conv2d(x, W, strides)
    if bias_term:
        res += bias_variable([n])
    return res

def dense(x, n):
    W, b = weight_variable([volume(x), n]), bias_variable([n])
    return tf.matmul(x, W) + b

def activation(x):
    print "ReLU"
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
    print "Batch Norm"
    eps = 1e-5
    gamma = tf.Variable(tf.constant(1.0, shape=[channels(x)]))
    beta = tf.Variable(tf.constant(0.0, shape=[channels(x)]))
    mean, variance = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps) 

def accuracy_score(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
    return accuracy

