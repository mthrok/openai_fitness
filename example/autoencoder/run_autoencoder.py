"""Train Autoencoder on MNIST dataset"""
from __future__ import division
from __future__ import print_function

import os
import gzip
import pickle
import logging
import argparse

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
import luchador.nn as nn


def _parase_command_line_args():
    default_mnist_path = os.path.join('data', 'mnist.pkl.gz')
    default_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'autoencoder.yml'
    )

    parser = argparse.ArgumentParser(
        description='Test autoencoder'
    )
    parser.add_argument(
        '--model', default=default_model_file,
        help=(
            'Model configuration file which contains Autoencoder. '
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
            'train': train_set[0].reshape(shape),
            'test': test_set[0].reshape(shape),
            'valid': valid_set[0].reshape(shape),
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


def _build_model(model_file, input_shape):
    model_def = nn.get_model_config(model_file, input_shape=input_shape)
    return nn.make_model(model_def)


def _build_cost(autoencoder):
    sse = nn.get_cost('SSE')()
    return sse(target=autoencoder.input, prediction=autoencoder.output)


def _build_optimization(autoencoder, cost):
    optimizer = nn.get_optimizer('Adam')(learning_rate=0.01)
    wrt = autoencoder.get_parameters_to_train()
    minimize_op = optimizer.minimize(loss=cost, wrt=wrt)
    update_op = autoencoder.get_update_operations()
    return update_op + [minimize_op]


def _train(session, autoencoder, cost, updates, images, batch_size):
    n_images = len(images)
    n_batch = n_images//batch_size
    train_data = images[:batch_size * n_batch]

    print('{:>6s}: {}'.format('Batch', 'Cost'))
    cst = 0
    for i in range(n_batch):
        batch = train_data[i*batch_size:(i+1)*batch_size, ...]
        cost_ = session.run(
            inputs={autoencoder.input: batch},
            outputs=cost, updates=updates, name='opt',
        )
        cst += cost_.tolist()
        if i and i % 100 == 0:
            print('{:>6d}: {:>8.3e}'.format(i, cst / 100))
            cst = 0


def _tile(image):
    n_img, h, w = image.shape[:3]
    n_tile = int(np.ceil(np.sqrt(n_img)))
    img = np.zeros((n_tile * h, n_tile * w), dtype='uint8')
    for i in range(n_img):
        row, col = i // n_tile, i % n_tile
        img[h*row:h*(row+1), w*col:w*(col+1)] = image[i]
    return img


def _plot(original, recon):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axis = fig.add_subplot(1, 2, 1)
    axis.imshow(_tile(original), cmap='gray')
    axis.set_title('Original Images')
    axis = fig.add_subplot(1, 2, 2)
    axis.imshow(_tile(recon), cmap='gray')
    axis.set_title('Reconstructed Images')
    print('Plot ready')
    plt.show()


def _main():
    args = _parase_command_line_args()
    _initialize_logger(args.debug)

    data_format = luchador.get_nn_conv_format()
    batch_size = 32
    input_shape = (
        [batch_size, 28, 28, 1] if data_format == 'NHWC' else
        [batch_size, 1, 28, 28]
    )

    autoencoder = _build_model(args.model, input_shape)
    cost = _build_cost(autoencoder)
    updates = _build_optimization(autoencoder, cost)
    images = _load_data(args.mnist, data_format)

    session = nn.Session()
    session.initialize()

    summary = nn.SummaryWriter(output_dir='tmp')
    if session.graph:
        summary.add_graph(session.graph)

    try:
        _train(
            session, autoencoder, cost, updates, images['train'], batch_size)
    except KeyboardInterrupt:
        pass

    orig = images['test'][:batch_size, ...]
    recon = session.run(
        outputs=autoencoder.output,
        inputs={autoencoder.input: orig}
    )

    axis = 3 if data_format == 'NHWC' else 1
    original = 255 * np.squeeze(orig, axis=axis)
    recon = 255 * np.squeeze(recon, axis=axis)

    if not args.no_plot:
        _plot(original.astype('uint8'), recon.astype('uint8'))


if __name__ == '__main__':
    _main()
