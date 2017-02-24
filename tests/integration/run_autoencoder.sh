#!/bin/bash
set -eu

mkdir -p data
DATA='data/mnist.pkl.gz'

if [ ! -f "${DATA}" ]; then
    echo "Downloading MNIST"
    curl -o "${DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi
python example/autoencoder/run_autoencoder.py --mnist "${DATA}" --no-plot
