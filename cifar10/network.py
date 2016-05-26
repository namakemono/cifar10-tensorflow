import numpy as np
import tensorflow as tf
import functions as F
from sklearn.metrics import accuracy_score

class BaseCifar10Classifier(object):
    def __init__(self, image_size=24, num_classes=10, batch_size=50, channels=3):
        self._image_size = image_size
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._channels = channels
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._images = tf.placeholder(tf.float32, shape=[None, self._image_size, self._image_size, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") 
        self._logits = self._inference(self._images, self._keep_prob)
        self._avg_loss = self._loss(self._labels, self._logits)
        self._train_op = self._train(self._avg_loss)
        self._accuracy = F.accuracy_score(self._labels, self._logits)
        self._saver = tf.train.Saver(tf.all_variables())
        self._session.run(tf.initialize_all_variables())

    def fit(self, X, y, max_epoch = 10):
        for epoch in range(max_epoch):
            for i in range(0, len(X), self._batch_size):
                batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]
                feed_dict={self._images: batch_images, self._labels: batch_labels, self._keep_prob: 0.5}
                _, train_avg_loss, global_step = self._session.run(fetches=[self._train_op, self._avg_loss, self._global_step], feed_dict=feed_dict)
            print "epochs =", global_step

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
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

    def save(self, filepath):
        self._saver.save(self._session, filepath)

    def _inference(self, X, keep_prob):
        pass

    def _loss(self, labels, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        # avg_loss = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        # tf.add_to_collection('losses', avg_loss)
        return tf.add_n(tf.get_collection('losses'))

    def _train(self, avg_loss):
        return tf.train.AdamOptimizer().minimize(avg_loss, self._global_step)

class Cifar10Classifier_01(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = F.dense(h, self._num_classes)
        return h

class Cifar10Classifier_02(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h

class Cifar10Classifier_03(BaseCifar10Classifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(X, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h

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
        return h

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
        return h

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
        return h

class Cifar10Classifier_ResNet(BaseCifar10Classifier):
    def __init__(self, layers):
        self._layers = layers
        super(Cifar10Classifier_ResNet, self).__init__(image_size=24, batch_size=128)

    def _residual(self, h, channels, strides):
        pass

    def _inference(self, X, keep_prob):
        h = X
        h = F.activation(F.batch_normalization(F.conv(h, 16)))
        for i in range(self._layers):
            h0 = h
            h1 = F.activation(F.batch_normalization(F.conv(h0, 16)))
            h2 = F.batch_normalization(F.conv(h1, 16))
            h = h2 + h0
            # c.f. http://gitxiv.com/comments/7rffyqcPLirEEsmpX
        for channels in [32, 64]:
            for i in range(self._layers):        
                h0 = h
                strides = 2 if i == 0 else 1
                h1 = F.activation(F.batch_normalization(F.conv(h0, channels, strides)))
                h2 = F.batch_normalization(F.conv(h1, channels))
                if F.volume(h0) == F.volume(h2):
                    h = h2 + h0
                else:
                    h3 = F.avg_pool(h0)
                    h4 = tf.pad(h3, [[0,0], [0,0], [0,0], [channels / 4, channels / 4]])
                    h = h2 + h4
        h = tf.reduce_mean(h, reduction_indices=[1, 2]) # Global Average Pooling
        h = F.dense(h, self._num_classes)
        return h
   
    def _train(self, avg_loss):
        lr = tf.select(tf.less(self._global_step, 32000), 0.1, tf.select(tf.less(self._global_step, 48000), 0.01, 0.001))
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(avg_loss, global_step=self._global_step)

class Cifar10Classifier_ResNet20(Cifar10Classifier_ResNet):
    def __init__(self):
        super(Cifar10Classifier_ResNet20, self).__init__(layers=3)

class Cifar10Classifier_ResNet32(Cifar10Classifier_ResNet):
    def __init__(self):
        super(Cifar10Classifier_ResNet32, self).__init__(layers=5)

class Cifar10Classifier_ResNet44(Cifar10Classifier_ResNet):
    def __init__(self):
        super(Cifar10Classifier_ResNet44, self).__init__(layers=7)

class Cifar10Classifier_ResNet56(Cifar10Classifier_ResNet):
    def __init__(self):
        super(Cifar10Classifier_ResNet56, self).__init__(layers=9)

class Cifar10Classifier_ResNet110(Cifar10Classifier_ResNet):
    def __init__(self):
        super(Cifar10Classifier_ResNet110, self).__init__(layers=18)


