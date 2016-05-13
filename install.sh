#!/bin/bash

curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
 && tar xvzfo cifar-10-python.tar.gz \
 && mkdir -p input \
 && mv cifar-10-batches-py input/ \
 && rm cifar-10-python.tar.gz
