import numpy as np
import tensorflow as tf
import functions as F
import gentleman
from sklearn.metrics import accuracy_score

class BaseCifar10Classifier(object):
    def __init__(self):
        self._image_size = 24
        self._num_classes = 10
        self._batch_size = 50
        self._epoch = 1
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gentleman.request_mem(6*1024, i_am_nice=False))
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._images = tf.placeholder("float", shape=[None, self._image_size, self._image_size, 3])
        self._labels = tf.placeholder("float", shape=[None, self._num_classes])
        self._keep_prob = tf.placeholder("float")
        self._global_step = tf.placeholder("int32") 
        self._logits = self._inference(self._images, self._keep_prob)
        self._avg_loss = self._loss(self._labels, self._logits)
        self._train_op = self._train(self._avg_loss)
        self._accuracy = F.accuracy_score(self._labels, self._logits)
        self._session.run(tf.initialize_all_variables())

    def fit(self, X, y, max_epoch = 10):
        for epoch in range(max_epoch):
            for i in range(0, len(X), self._batch_size):
                batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]
                feed_dict={self._images: batch_images, self._labels: batch_labels, self._keep_prob: 0.5, self._global_step: self._epoch}
                _, train_avg_loss = self._session.run(fetches=[self._train_op, self._avg_loss], feed_dict=feed_dict)
            self._epoch += 1

    def predict(self, X):
        res = None
        for i in range(0, len(X), self._batch_size):
            batch_images = X[i:i+self._batch_size]
            feed_dict={self._images: batch_images, self._keep_prob: 1.0}
            test_logits = self._session.run(fetches=self._logits, feed_dict=feed_dict)
            if res is None:
                res = np.argmax(test_logits, axis=1)
            else:
                res = np.r_[res, np.argmax(test_logits, axis=1)]
        return res
    
    def predict_proba(self, X):
        res = None
        for i in range(0, len(X), self._batch_size):
            batch_images = X[i:i+self._batch_size]
            feed_dict={self._images: batch_images, self._keep_prob: 1.0}
            test_logits = self._session.run(fetches=self._logits, feed_dict=feed_dict)
            if res is None:
                res = test_logits
            else:
                res = np.r_[res, test_logits]
        return res

    def score(self, X, y):
        acc_list, total_loss = [], 0
        for i in range(0, len(X), self._batch_size):
            batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]
            feed_dict={self._images: batch_images, self._labels: batch_labels, self._keep_prob: 1.0}
            acc, avg_loss = self._session.run(fetches=[self._accuracy, self._avg_loss], feed_dict=feed_dict)
            acc_list.append(acc)
            total_loss += avg_loss * len(batch_images)
        return np.asarray(acc_list).mean(), total_loss / len(X)

    def _inference(self, X, keep_prob):
        pass

    def _loss(self, labels, logits):
        return -tf.reduce_mean(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

    def _train(self, avg_loss):
        return tf.train.AdamOptimizer(1e-4).minimize(avg_loss)

class Cifar10Classifier_01(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

class Cifar10Classifier_02(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

class Cifar10Classifier_03(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(X, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

class Cifar10Classifier_04(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(X, 64))
        h = F.activation(F.conv(X, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

class Cifar10Classifier_05(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(h, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.activation(F.conv(h, 128))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h) 

class Cifar10Classifier_06(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(h, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.activation(F.conv(h, 128))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.activation(F.dense(h, 256))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

class Cifar10Classifier_07(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = tf.nn.lrn(h, 4)
        h = F.activation(F.conv(h, 64))
        h = F.activation(F.conv(h, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = tf.nn.lrn(h, 4)
        h = F.activation(F.conv(h, 128))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = tf.nn.lrn(h, 4)
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.activation(F.dense(h, 256))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

class Cifar10Classifier_ResNet20(BaseCifar10Classifier):
    def __init__(self):
        self._layers = 3
        super(Cifar10Classifier_ResNet20, self).__init__()

    def _inference(self, X, keep_prob):
        h = X
        for channels in [16, 32, 64]:
            h = F.activation(F.batch_normalization(F.conv(h, channels)))
            for i in range(self._layers):
                h = F.residual(h, channels)
        h = F.avg_pool(h, ksize=h.get_shape()[1], strides=h.get_shape()[1])
        h = F.flatten(h)
        h = F.dense(h, self._num_classes)
        return tf.nn.softmax(h)

