#!/bin/bash
set -eux

DATA="${HOME}/.mnist/mnist.pkl.gz"
if [ ! -f "${DATA}" ]; then
    echo "Downloading MNIST"
    curl --create-dirs -o "${DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi

python example/classification/classify_mnist.py --mnist "${DATA}" --output tmp/mnist  --model example/classification/model.yml
python example/autoencoder/train_ae.py          --mnist "${DATA}" --output tmp/ae/ae  --model example/autoencoder/autoencoder.yml
python example/autoencoder/train_ae.py          --mnist "${DATA}" --output tmp/ae/vae --model example/autoencoder/variational_autoencoder.yml
python example/gan/train_gan.py                 --mnist "${DATA}" --output tmp/gan    --model example/gan/model.yml
