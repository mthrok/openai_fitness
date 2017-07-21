#!/bin/bash
set -eux

MOCK='';
if [ "${CIRCLECI:-false}" = true ];
then
    MOCK='--mock';
fi

python example/classification/classify_mnist.py ${MOCK} --model example/classification/model.yml
python example/autoencoder/train_ae.py          ${MOCK} --model example/autoencoder/autoencoder.yml
python example/autoencoder/train_vae.py         ${MOCK} --model example/autoencoder/variational_autoencoder.yml
python example/gan/train_gan.py                 ${MOCK} --model example/gan/gan.yml
python example/gan/train_dcgan.py               ${MOCK} --model example/gan/dcgan.yml --n-iterations 10
