"""Train Autoencoder on MNIST dataset"""
from __future__ import division

import os
import gzip
import pickle
import logging
import argparse

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
import luchador.nn as nn

_LG = logging.getLogger('luchador')


def _parase_command_line_args():
    default_mnist_path = os.path.join('data', 'mnist.pkl.gz')
    default_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'classifier.yml'
    )

    parser = argparse.ArgumentParser(
        description='Test autoencoder'
    )
    parser.add_argument(
        '--model', default=default_model_file,
        help=(
            'Model configuration file which contains classifier. '
            'Default: {}'.format(default_model_file)
        )
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
    parser.add_argument('--no-plot', action='store_true')
    return parser.parse_args()


def _load_data(filepath, data_format):
    with gzip.open(filepath, 'rb') as file_:
        train_set, test_set, valid_set = pickle.load(file_)
        shape = [-1, 28, 28, 1] if data_format == 'NHWC' else [-1, 1, 28, 28]
        return {
            'train': {
                'data': train_set[0].reshape(shape),
                'label': train_set[1],
            },
            'test': {
                'data': test_set[0].reshape(shape),
                'label': test_set[1],
            },
            'valid': {
                'data': valid_set[0].reshape(shape),
                'label': valid_set[1],
            },
        }


def _initialize_logger(debug):
    from luchador.util import initialize_logger
    message_format = (
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
        if debug else '%(asctime)s: %(levelname)5s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    initialize_logger(
        name='luchador', message_format=message_format, level=level)


def _build_model(model_file, input_shape, batch_size):
    model_def = nn.get_model_config(
        model_file, input_shape=input_shape,
        batch_size=batch_size, n_classes=10)
    return nn.make_model(model_def)


def _train(session, classifier, dataset, batch_size):
    data, label = dataset['data'], dataset['label']
    n_images = len(data)
    n_batch = n_images//batch_size
    data = data[:batch_size * n_batch]
    label = label[:batch_size * n_batch]

    _LG.info('%6s: %s', 'Batch', 'Cost')
    cst = 0
    for i in range(n_batch):
        data_batch = data[i*batch_size:(i+1)*batch_size, ...]
        label_batch = label[i*batch_size:(i+1)*batch_size, ...]
        cost_ = session.run(
            inputs={
                classifier.input['data']: data_batch,
                classifier.input['label']: label_batch,
            },
            outputs=classifier.output['error'],
            updates=classifier.get_update_operations(),
            name='opt',
        )
        cst += cost_.tolist()
        if i and i % 100 == 0:
            _LG.info('%6d: %8.3f', i, cst / 100)
            cst = 0


def _test(session, classifier, dataset, batch_size):
    data, label = dataset['data'], dataset['label']
    n_images = len(data)
    n_batch = n_images//batch_size
    data = data[:batch_size * n_batch]
    label = label[:batch_size * n_batch]

    cst = 0
    for i in range(n_batch):
        data_batch = data[i*batch_size:(i+1)*batch_size, ...]
        label_batch = label[i*batch_size:(i+1)*batch_size, ...]
        cost_ = session.run(
            inputs={
                classifier.input['data']: data_batch,
                classifier.input['label']: label_batch,
            },
            outputs=classifier.output['error'],
            name='test',
        )
        cst += cost_.tolist() / n_batch
    _LG.info('Test Error %s', cst)


def _main():
    args = _parase_command_line_args()
    _initialize_logger(args.debug)

    data_format = luchador.get_nn_conv_format()
    batch_size = 32
    input_shape = (
        [batch_size, 28, 28, 1] if data_format == 'NHWC' else
        [batch_size, 1, 28, 28]
    )

    classifier = _build_model(args.model, input_shape, batch_size)
    dataset = _load_data(args.mnist, data_format)

    session = nn.Session()
    session.initialize()

    summary = nn.SummaryWriter(output_dir='tmp')
    if session.graph:
        summary.add_graph(session.graph)

    try:
        _train(session, classifier, dataset['train'], batch_size)
        _test(session, classifier, dataset['test'], batch_size)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    _main()
