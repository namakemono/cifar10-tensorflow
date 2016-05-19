import os
import time, datetime
import numpy as np
import pandas as pd
from network import *
import datasets

def run():
    test_images, test_labels = datasets.load_cifar10(is_train=False)
    for clf in [Cifar10Classifier_ResNet110()]: # , Cifar10Classifier_ResNet44(), Cifar10Classifier_ResNet56(), Cifar10Classifier_ResNet110()]:
        records = []
        for epoch in range(200):
            train_images, train_labels = datasets.load_cifar10(is_train=True)
            num_epoch = 1
            start_time = time.time()
            clf.fit(train_images, train_labels, max_epoch=num_epoch)
            duration = time.time() - start_time
            examples_per_sec = (num_epoch * len(train_images)) / duration
            train_accuracy, train_loss =  clf.score(train_images, train_labels)
            test_accuracy, test_loss = clf.score(test_images, test_labels)
            summary = {
                "epoch": epoch,
                "name": clf.__class__.__name__,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "examples_per_sec": examples_per_sec,
            }
            print "[%(epoch)d][%(name)s]train-acc: %(train_accuracy).3f, train-loss: %(train_loss).3f," % summary,
            print "test-acc: %(test_accuracy).3f, test-loss: %(test_loss).3f, %(examples_per_sec).1f examples/sec" % summary 
            records.append(summary)
            df = pd.DataFrame(records)
            df.to_csv("../output/%s.csv" % clf.__class__.__name__.lower(), index=False)
            if df["test_accuracy"].max() - 1e-5 < test_accuracy:
                save_dir = "../models/%s" % clf.__class__.__name__
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                print "Save to %s" % save_dir
                clf.save(save_dir + "/model.ckpt")
            if train_loss * 30 < test_loss: # Overfitting
                break

if __name__ == "__main__":
    run()

