#!/bin/bash
set -eu

mkdir -p data
DATA='data/mnist.pkl.gz'

if [ ! -f "${DATA}" ]; then
    echo "Downloading MNIST"
    curl -o "${DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi
python example/classification/classify_mnist.py --mnist "${DATA}" --model example/classification/classifier.yml
