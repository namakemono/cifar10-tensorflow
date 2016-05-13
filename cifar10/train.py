import os
import time, datetime
import numpy as np
import pandas as pd
from network import *
import datasets

def run():
    test_images, test_labels = datasets.load_cifar10(is_train=False)
    for clf in [Cifar10Classifier_07()]:
        records = []
        for epoch in range(1000):
            train_images, train_labels = datasets.load_cifar10(is_train=True)
            clf.fit(train_images, train_labels, max_epoch=1)
            train_accuracy, train_loss =  clf.score(train_images, train_labels)
            test_accuracy, test_loss = clf.score(test_images, test_labels)
            summary = {
                "epoch": epoch,
                "name": clf.__class__.__name__,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
            print "[%(epoch)d][%(name)s]train-acc: %(train_accuracy).3f, train-loss: %(train_loss).3f, test-acc: %(test_accuracy).3f, test-loss: %(test_loss).3f" % summary 
            records.append(summary)
            pd.DataFrame(records).to_csv("../output/%s.csv" % clf.__class__.__name__.lower(), index=False)
            if train_loss * 30 < test_loss: # Overfitting
                break
 
if __name__ == "__main__":
    run()

