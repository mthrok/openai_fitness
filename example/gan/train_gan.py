import gzip
import pickle
import os.path
import logging

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from luchador import nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_LG = logging.getLogger(__name__)


def _parse_command_line_args():
    import argparse
    default_mnist_path = os.path.join(os.path.expanduser('~'), '.mnist')
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


def _initialize_logger(debug):
    from luchador.util import initialize_logger
    message_format = (
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
        if debug else '%(asctime)s: %(levelname)5s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    initialize_logger(
        name='luchador', message_format=message_format, level=level)


def _build_model(model_file):
    _LG.info('Loading model %s', model_file)
    model_def = nn.get_model_config(model_file)
    return nn.make_model(model_def)



def _load_data(filepath):
    return input_data.read_data_sets(filepath, one_hot=True)


def _sample_seed(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def _train(optimize_disc, optimize_gen, generate_image):
    def plot(block):
        images = generate_image()
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(images):
            ax = fig.add_subplot(gs[i])
            ax.imshow(sample, cmap='Greys_r')
        plt.show(block=block)
    for i in range(10):
        plot(False)
        for _ in range(1000):
            disc_loss = optimize_disc()
            gen_loss = optimize_gen()
        print i, disc_loss, gen_loss
    plot(True)


def _main():
    args = _parse_command_line_args()
    _initialize_logger(args.debug)

    generator = _build_model(args.generator)
    gen_seed = nn.Input(shape=(None, 100))
    in_gen = generator(gen_seed)
    in_real = nn.Input(shape=(None, 784))

    discriminator = _build_model(args.discriminator)

    logit_real = discriminator(in_real)
    logit_fake = discriminator(in_gen)

    sce_gen = nn.cost.SigmoidCrossEntropy(scope='sce_gen')
    sce_real = nn.cost.SigmoidCrossEntropy(scope='sce_real')
    sce_fake = nn.cost.SigmoidCrossEntropy(scope='sce_fake')
    gen_loss = sce_gen(prediction=logit_fake, target=1)

    disc_loss_real = sce_real(prediction=logit_real, target=1)
    disc_loss_fake = sce_fake(prediction=logit_fake, target=0)
    disc_loss = disc_loss_real + disc_loss_fake

    optimizer_disc = nn.optimizer.Adam(
        learning_rate=0.001, scope='TrainDiscriminator/Adam')
    optimizer_gen = nn.optimizer.Adam(
        learning_rate=0.001, scope='TrainGenerator/Adam')

    opt_disc = optimizer_disc.minimize(
        disc_loss, discriminator.get_parameters_to_train())
    opt_gen = optimizer_gen.minimize(
        gen_loss, generator.get_parameters_to_train())

    dataset = _load_data(args.mnist)

    batch_size = 32
    sess = nn.Session()
    sess.initialize()

    def optimize_disc():
        return sess.run(
            inputs={
                gen_seed: _sample_seed(batch_size, 100),
                in_real: dataset.train.next_batch(batch_size)[0],
            },
            outputs=disc_loss,
            updates=opt_disc,
        )

    def optimize_gen():
        return sess.run(
            inputs={
                gen_seed: _sample_seed(batch_size, 100),
            },
            outputs=gen_loss,
            updates=opt_gen,
        )

    def generate_image():
        return sess.run(
            inputs={
                gen_seed: _sample_seed(16, 100),
            },
            outputs=in_gen,
        ).reshape(-1, 28, 28)

    _train(optimize_disc, optimize_gen, generate_image)


if __name__ == '__main__':
    _main()
