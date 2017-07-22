"""Train DCGAN on Face dataset"""
from __future__ import division
from __future__ import absolute_import

import os
import logging

import numpy as np
import luchador
from luchador import nn

from example.utils import (
    initialize_logger, load_celeba_face,
)

_LG = logging.getLogger(__name__)


def _parse_command_line_args():
    import argparse
    default_data_path = os.path.join(
        os.path.expanduser('~'), '.dataset', 'celeba_faces.pkl.gz')
    default_model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dcgan.yml'
    )
    parser = argparse.ArgumentParser(
        description=(
            'Run Deep Conovolutional Generative Adversarial Network '
            'on face dataset.'
        )
    )
    parser.add_argument(
        '--model', default=default_model_file,
        help=(
            'Generator model configuration file. '
            'Default: {}'.format(default_model_file)
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
        '--dataset', default=default_data_path,
        help=(
            'Path to pickled CelebA face dataset.'
            'Default: {}'.format(default_data_path)
        ),
    )
    parser.add_argument(
        '--output',
        help=(
            'When provided, plot generated data to this directory.'
        )
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Mock test data to run the script without data for testing.'
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def _build_models(model_file):
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
        learning_rate=0.0002, beta1=0.5, scope='TrainDiscriminator/Adam')
    optimizer_gen = nn.optimizer.Adam(
        learning_rate=0.0002, beta1=0.5, scope='TrainGenerator/Adam')

    opt_gen = optimizer_gen.minimize(
        gen_loss, generator.get_parameters_to_train())
    opt_disc = optimizer_disc.minimize(
        disc_loss, discriminator.get_parameters_to_train())
    return opt_gen, opt_disc


def _train(
        train_disc, train_gen, summarize,
        n_iterations, n_epochs):
    _LG.info('%5s: %10s, %10s', 'EPOCH', 'DISC_LOSS', 'GEN_LOSS')
    summarize(0)
    for epoch in range(1, n_epochs+1):
        disc_loss, gen_loss = 0.0, 0.0
        for _ in range(n_iterations):
            disc_loss += train_disc() / n_iterations
            gen_loss += train_gen() / n_iterations
        _LG.info('%5s: %10.3e, %10.3e', epoch, disc_loss, gen_loss)
        summarize(epoch, (disc_loss, gen_loss))


def _sample_seed(*size):
    return np.random.normal(size=size).astype('float32')


def _main():
    args = _parse_command_line_args()
    initialize_logger(args.debug)

    batch_size = 32
    format_ = luchador.get_nn_conv_format()
    dataset = load_celeba_face(
        args.dataset, data_format=format_, mock=args.mock)

    model = _build_models(args.model)
    discriminator, generator = model['discriminator'], model['generator']

    input_gen = nn.Input(shape=(None, args.n_seeds), name='GeneratorInput')
    data_shape = (None,) + dataset.train.shape[1:]
    data_real = nn.Input(shape=data_shape, name='InputData')
    _LG.info('Building Generator')
    data_fake = generator(input_gen)

    _LG.info('Building fake discriminator')
    logit_fake = discriminator(data_fake)
    _LG.info('Building real discriminator')
    logit_real = discriminator(data_real)

    gen_loss, disc_loss = _build_loss(logit_real, logit_fake)
    opt_gen, opt_disc = _build_optimization(
        generator, gen_loss, discriminator, disc_loss)

    sess = nn.Session()
    sess.initialize()

    _summary_writer = None
    if args.output:
        _summary_writer = nn.SummaryWriter(output_dir=args.output)
        if sess.graph is not None:
            _summary_writer.add_graph(sess.graph)

    def _train_disc():
        return sess.run(
            inputs={
                input_gen: _sample_seed(batch_size, args.n_seeds),
                data_real: dataset.train.next_batch(batch_size).data
            },
            outputs=disc_loss,
            updates=discriminator.get_update_operations() + [opt_disc],
            name='train_discriminator',
        )

    def _train_gen():
        return sess.run(
            inputs={
                input_gen: _sample_seed(batch_size, args.n_seeds),
            },
            outputs=gen_loss,
            updates=generator.get_update_operations() + [opt_gen],
            name='train_generator',
        )

    random_seed = _sample_seed(batch_size, args.n_seeds)
    def _summarize(epoch, losses=None):
        if not args.output:
            return

        if losses:
            _summary_writer.summarize(
                summary_type='scalar',
                global_step=epoch,
                dataset={
                    'Generator/Loss': losses[0],
                    'Discriminator/Loss': losses[1],
                },
            )

        images = sess.run(
            inputs={
                input_gen: random_seed,
            },
            outputs=data_fake,
            name='generate_samples',
        )
        if format_ == 'NCHW':
            images = images.transpose(0, 2, 3, 1)
        images = (255 * images).astype(np.uint8)
        _summary_writer.summarize(
            summary_type='image',
            global_step=epoch,
            dataset={
                'Genearated/epoch_{:02d}'.format(epoch): images
            },
            max_outputs=10,
        )

    _train(
        _train_disc, _train_gen, _summarize,
        args.n_iterations, args.n_epochs,
    )


if __name__ == '__main__':
    _main()
