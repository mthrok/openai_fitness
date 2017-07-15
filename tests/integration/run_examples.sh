#!/bin/bash
set -eux

DATA="${HOME}/.mnist/mnist.pkl.gz"
if [ ! -f "${DATA}" ]; then
    echo "Downloading MNIST"
    curl --create-dirs -o "${DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi

python example/classification/classify_mnist.py --mnist "${DATA}" --model example/classification/model.yml
python example/autoencoder/train_ae.py          --mnist "${DATA}" --model example/autoencoder/autoencoder.yml
python example/autoencoder/train_vae.py         --mnist "${DATA}" --model example/autoencoder/variational_autoencoder.yml
python example/gan/train_gan.py                 --mnist "${DATA}" --model example/gan/gan.yml
python example/gan/train_dcgan.py               --mnist "${DATA}" --model example/gan/dcgan.yml --n-iterations 100
