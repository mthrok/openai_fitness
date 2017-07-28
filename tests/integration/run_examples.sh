#!/bin/bash
set -eux

MOCK='';
if [ "${CIRCLECI:-false}" = true ];
then
    MOCK='--mock';
fi

if [ "${COUNT_INTEGRATION_COVERAGE:-false}" = true ]; then
    TEST_COMMAND="coverage run"
else
    TEST_COMMAND="python"
fi

${TEST_COMMAND} example/classification/classify_mnist.py ${MOCK} --model example/classification/model.yml
${TEST_COMMAND} example/autoencoder/train_ae.py          ${MOCK} --model example/autoencoder/autoencoder.yml
${TEST_COMMAND} example/autoencoder/train_vae.py         ${MOCK} --model example/autoencoder/variational_autoencoder.yml
${TEST_COMMAND} example/gan/train_gan.py                 ${MOCK} --model example/gan/gan.yml
${TEST_COMMAND} example/gan/train_dcgan.py               ${MOCK} --model example/gan/dcgan.yml --n-iterations 10
