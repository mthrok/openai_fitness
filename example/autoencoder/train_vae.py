"""Train Autoencoder on MNIST dataset"""
from __future__ import division
from __future__ import absolute_import

import os
import logging

import numpy as np
import luchador
import luchador.nn as nn

from example.utils import (
    plot_images, initialize_logger, load_mnist
)

_LG = logging.getLogger(__name__)


def _parase_command_line_args():
    import argparse
    default_mnist_path = os.path.join(
        os.path.expanduser('~'), '.mnist', 'mnist.pkl.gz')
    default_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'variational_autoencoder.yml'
    )

    parser = argparse.ArgumentParser(
        description='Train autoencoder on MNIST and reconstruct images.'
    )
    parser.add_argument(
        '--model', default=default_model_file,
        help=(
            'Model configuration file which contains Autoencoder. '
            'Default: {}'.format(default_model_file)
        )
    )
    parser.add_argument(
        '--output',
        help=(
            'When provided, plot generated data to this directory.'
        )
    )
    parser.add_argument(
        '--n-iterations', default=100, type=int,
        help='#Trainingss par epoch.'
    )
    parser.add_argument(
        '--n-epochs', default=10, type=int,
        help='#Epochs to run.'
    )
    parser.add_argument(
        '--mnist', default=default_mnist_path,
        help=(
            'Path to MNIST dataset, downloaded from '
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz '
            'Default: {}'.format(default_mnist_path)
        ),
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _build_model(model_file, data_format, batch_size):
    input_shape = (
        [batch_size, 28, 28, 1] if data_format == 'NHWC' else
        [batch_size, 1, 28, 28]
    )
    model_def = nn.get_model_config(model_file, input_shape=input_shape)
    return nn.make_model(model_def)


def _train(train_ae, plot_reconstruction, n_iterations=100, n_epochs=10):
    plot_reconstruction(0)
    _LG.info(
        '%5s: %12s %12s %12s',
        'EPOCH', 'RECON_LOSS', 'LATENT_LOSS', 'TOTAL_LOSS')
    for epoch in range(1, n_epochs+1):
        loss = np.asarray([0.0, 0.0, 0.0])
        for _ in range(n_iterations):
            loss += train_ae()
        plot_reconstruction(epoch)
        loss /= n_iterations
        _LG.info(
            '%5d: %12.2e %12.2e %12.2e', epoch, loss[0], loss[1], loss[2])


def _main():
    args = _parase_command_line_args()
    initialize_logger(args.debug)

    batch_size = 32
    data_format = luchador.get_nn_conv_format()
    autoencoder = _build_model(args.model, data_format, batch_size)
    mnist = load_mnist(args.mnist, data_format=data_format)

    sess = nn.Session()
    sess.initialize()

    if args.output:
        summary = nn.SummaryWriter(output_dir=args.output)
        if sess.graph is not None:
            summary.add_graph(sess.graph)

    def _train_ae():
        batch = mnist.train.next_batch(batch_size).data
        return sess.run(
            inputs={autoencoder.input: batch},
            outputs=autoencoder.output['error'],
            updates=autoencoder.get_update_operations(),
            name='train_autoencoder',
        )

    def _plot_reconstruction(epoch):
        if not args.output:
            return
        orig = mnist.test.next_batch(batch_size).data
        recon = sess.run(
            inputs={autoencoder.input: orig},
            outputs=autoencoder.output['reconstruction'],
            name='reconstruct_images',
        )
        axis = 3 if data_format == 'NHWC' else 1
        orig = np.squeeze(orig, axis=axis)
        recon = np.squeeze(recon, axis=axis)

        base_path = os.path.join(args.output, '{:03}_'.format(epoch))
        plot_images(orig, base_path + 'orign.png')
        plot_images(recon, base_path + 'recon.png')

    _train(
        _train_ae, _plot_reconstruction,
        n_iterations=args.n_iterations, n_epochs=args.n_epochs
    )


if __name__ == '__main__':
    _main()
