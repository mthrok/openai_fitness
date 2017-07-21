#!/bin/bash
set -eux

MNIST_DATA="${HOME}/.dataset/mnist.pkl.gz"
if [ ! -f "${MNIST_DATA}" ]; then
    echo "Downloading MNIST"
    curl --create-dirs -o "${MNIST_DATA}" http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
fi

FACE_DATA="${HOME}/.dataset/celeba_faces_test.pkl.gz"
if [ ! -f "${FACE_DATA}" ]; then
    echo "Downloading Face dataset"
    curl --create-dirs -o "${FACE_DATA}" https://s3.amazonaws.com/luchador/dataset/celeba/celeba_faces_test.pkl.gz
fi

python example/classification/classify_mnist.py --mnist "${MNIST_DATA}" --model example/classification/model.yml
python example/autoencoder/train_ae.py          --mnist "${MNIST_DATA}" --model example/autoencoder/autoencoder.yml
python example/autoencoder/train_vae.py         --mnist "${MNIST_DATA}" --model example/autoencoder/variational_autoencoder.yml
python example/gan/train_gan.py                 --mnist "${MNIST_DATA}" --model example/gan/gan.yml
python example/gan/train_dcgan.py               --mnist "${FACE_DATA}"  --model example/gan/dcgan.yml --n-iterations 10
