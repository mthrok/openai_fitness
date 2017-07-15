#!/bin/bash
set -eu

DATA="${HOME}/.mnist/mnist.pkl.gz"
if [ ! -f "${DATA}" ]; then
    echo "Downloading MNIST"
    curl --create-dirs -o "${DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi
python example/autoencoder/run_autoencoder.py --mnist "${DATA}" --model example/autoencoder/autoencoder.yml --output tmp/ae/ae
python example/autoencoder/run_autoencoder.py --mnist "${DATA}" --model example/autoencoder/variational_autoencoder.yml --output tmp/ae/vae
