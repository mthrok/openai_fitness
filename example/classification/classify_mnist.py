"""Train Autoencoder on MNIST dataset"""
from __future__ import division

import os
import logging

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
import luchador.nn as nn

from example.utils import initialize_logger, load_mnist

_LG = logging.getLogger(__name__)


def _parase_command_line_args():
    import argparse
    default_mnist_path = os.path.join(
        os.path.expanduser('~'), '.dataset', 'mnist.pkl.gz')
    default_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'model.yml'
    )
    parser = argparse.ArgumentParser(
        description='Train MNIST classifier and test'
    )
    parser.add_argument(
        '--model', default=default_model_file,
        help=(
            'Model configuration file which contains classifier. '
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
        '--n-iterations', default=1000, type=int,
        help='#optimizatitons par epoch.'
    )
    parser.add_argument(
        '--n-epochs', default=10, type=int,
        help='#Epochs to run.'
    )
    parser.add_argument(
        '--dataset', default=default_mnist_path,
        help=(
            'Path to MNIST dataset, downloaded from '
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz '
            'Default: {}'.format(default_mnist_path)
        ),
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Mock test data to run the script without data for testing.'
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _build_model(model_file, data_format):
    input_shape = (
        (None, 28, 28, 1) if data_format == 'NHWC' else (None, 1, 28, 28)
    )
    model_def = nn.get_model_config(
        model_file, input_shape=input_shape, n_classes=10)
    return nn.make_model(model_def)


def _train(train_classifier, test_classifier, n_iterations=100, n_epochs=10):
    _LG.info('%5s: %10s %10s', 'EPOCH', 'TRAIN_LOSS', 'TEST_LOSS')
    for epoch in range(1, n_epochs+1):
        train_cost = 0.0
        for _ in range(n_iterations):
            train_cost += train_classifier() / n_iterations
        test_cost = test_classifier()
        _LG.info('%5d: %10.2e %10.2e', epoch, train_cost, test_cost)


def _main():
    args = _parase_command_line_args()
    initialize_logger(args.debug)

    batch_size = 32
    data_format = luchador.get_nn_conv_format()
    classifier = _build_model(args.model, data_format)
    dataset = load_mnist(args.dataset, data_format=data_format, mock=args.mock)

    sess = nn.Session()
    sess.initialize()

    if args.output:
        summary = nn.SummaryWriter(output_dir=args.output)
        if sess.graph is not None:
            summary.add_graph(sess.graph)

    def _train_classifier():
        batch = dataset.train.next_batch(batch_size)
        return sess.run(
            inputs={
                classifier.input['data']: batch.data,
                classifier.input['label']: batch.label,
            },
            outputs=classifier.output['error'],
            updates=classifier.get_update_operations(),
            name='train_classifier',
        )

    def _test_classifier():
        batch = dataset.test.next_batch(batch_size)
        return sess.run(
            inputs={
                classifier.input['data']: batch.data,
                classifier.input['label']: batch.label,
            },
            outputs=classifier.output['error'],
            name='test_classifier',
        )

    _train(
        _train_classifier, _test_classifier,
        n_iterations=args.n_iterations, n_epochs=args.n_epochs
    )


if __name__ == '__main__':
    _main()
