from __future__ import absolute_import

import os.path
import logging

import numpy as np
from luchador import nn

from example.utils import (
    initialize_logger, load_mnist, plot_images
)

_LG = logging.getLogger(__name__)


def _parse_command_line_args():
    import argparse
    default_mnist_path = os.path.join(
        os.path.expanduser('~'), '.mnist', 'mnist.pkl.gz')
    default_generator_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'generator.yml'
    )
    default_discriminator_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'discriminator.yml'
    )
    parser = argparse.ArgumentParser(
        description='Test Generative Adversarial Network'
    )
    parser.add_argument(
        '--generator', default=default_generator_file,
        help=(
            'Generator model configuration file. '
            'Default: {}'.format(default_generator_file)
        )
    )
    parser.add_argument(
        '--discriminator', default=default_discriminator_file,
        help=(
            'Generator model configuration file. '
            'Default: {}'.format(default_discriminator_file)
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
        '--n-seeds', default=100, type=int,
        help='#Generator input dimensions.'
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
    parser.add_argument(
        '--output',
        help=(
            'When provided, plot generated data to this directory.'
        )
    )
    return parser.parse_args()


def _build_model(model_file):
    _LG.info('Loading model %s', model_file)
    model_def = nn.get_model_config(model_file)
    return nn.make_model(model_def)


def _build_loss(logit_real, logit_fake):
    sce_gen = nn.cost.SigmoidCrossEntropy(scope='sce_gen')
    sce_real = nn.cost.SigmoidCrossEntropy(scope='sce_real')
    sce_fake = nn.cost.SigmoidCrossEntropy(scope='sce_fake')
    gen_loss = sce_gen(prediction=logit_fake, target=1)

    disc_loss_real = sce_real(prediction=logit_real, target=1)
    disc_loss_fake = sce_fake(prediction=logit_fake, target=0)
    disc_loss = disc_loss_real + disc_loss_fake
    return gen_loss, disc_loss


def _build_optimization(generator, gen_loss, discriminator, disc_loss):
    optimizer_disc = nn.optimizer.Adam(
        learning_rate=0.001, scope='TrainDiscriminator/Adam')
    optimizer_gen = nn.optimizer.Adam(
        learning_rate=0.001, scope='TrainGenerator/Adam')

    opt_gen = optimizer_gen.minimize(
        gen_loss, generator.get_parameters_to_train())
    opt_disc = optimizer_disc.minimize(
        disc_loss, discriminator.get_parameters_to_train())
    return opt_gen, opt_disc


def _train(
        optimize_disc, optimize_gen, generate_images,
        n_iterations, n_epochs, output=False):
    if output:
        path = os.path.join(output, '{:03d}.png'.format(0))
        plot_images(generate_images(), path)
    _LG.info('%5s: %10s, %10s', 'EPOCH', 'DISC_LOSS', 'GEN_LOSS')
    for i in range(n_epochs):
        for _ in range(n_iterations):
            disc_loss = optimize_disc()
            gen_loss = optimize_gen()
        _LG.info('%5s: %10.3e, %10.3e', i, disc_loss, gen_loss)
        if output:
            path = os.path.join(output, '{:03d}.png'.format(i))
            plot_images(generate_images(), path)


def _sample_seed(m, n):
    return np.random.uniform(-1., 1., size=[m, n]).astype('float32')


def _main():
    args = _parse_command_line_args()
    initialize_logger(args.debug)
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    batch_size = 32
    dataset = load_mnist(args.mnist, flatten=True)

    generator = _build_model(args.generator)
    discriminator = _build_model(args.discriminator)

    input_gen = nn.Input(shape=(None, args.n_seeds), name='GeneratorInput')
    data_real = nn.Input(shape=dataset.train.shape, name='InputData')
    data_fake = generator(input_gen)

    logit_fake = discriminator(data_fake)
    logit_real = discriminator(data_real)

    gen_loss, disc_loss = _build_loss(logit_real, logit_fake)
    opt_gen, opt_disc = _build_optimization(
        generator, gen_loss, discriminator, disc_loss)

    sess = nn.Session()
    sess.initialize()

    def _optimize_disc():
        return sess.run(
            inputs={
                input_gen: _sample_seed(batch_size, args.n_seeds),
                data_real: dataset.train.next_batch(batch_size).data
            },
            outputs=disc_loss,
            updates=opt_disc,
        )

    def _optimize_gen():
        return sess.run(
            inputs={
                input_gen: _sample_seed(batch_size, args.n_seeds),
            },
            outputs=gen_loss,
            updates=opt_gen,
        )

    def _generate_image():
        return sess.run(
            inputs={
                input_gen: _sample_seed(16, args.n_seeds),
            },
            outputs=data_fake,
        ).reshape(-1, 28, 28)

    _train(
        _optimize_disc, _optimize_gen, _generate_image,
        args.n_iterations, args.n_epochs, args.output)


if __name__ == '__main__':
    _main()
