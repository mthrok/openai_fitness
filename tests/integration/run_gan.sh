#!/bin/bash
set -eu

DATA="${HOME}/.mnist/mnist.pkl.gz"
if [ ! -f "${DATA}" ]; then
    echo "Downloading MNIST"
    curl --create-dirs -o "${DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi
python example/gan/train_gan.py --mnist "${DATA}"
