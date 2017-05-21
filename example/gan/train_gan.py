import gzip
import pickle
import os.path
import logging

from luchador import nn

_LG = logging.getLogger(__name__)


def _parse_command_line_args():
    import argparse
    default_mnist_path = os.path.join('data', 'mnist.pkl.gz')
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
    _LG.info('Loading data %s', filepath)
    with gzip.open(filepath, 'rb') as file_:
        train_set, test_set, valid_set = pickle.load(file_)
        shape = [-1, 28 * 28]
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


def _main():
    args = _parse_command_line_args()
    _initialize_logger(args.debug)

    generator = _build_model(args.generator)
    seed_gen = nn.Input(shape=(None, 100))
    in_gen = generator(seed_gen)
    in_real = nn.Input(shape=(None, 784))

    discriminator = _build_model(args.discriminator)

    logit_real = discriminator(in_real)
    logit_fake = discriminator(in_gen)

    sce_real = nn.cost.SigmoidCrossEntropy(scope='sce_real')
    sce_fake = nn.cost.SigmoidCrossEntropy(scope='sce_fake')
    disc_loss_real = sce_real(prediction=logit_real, target=1)
    disc_loss_fake = sce_fake(prediction=logit_fake, target=0)
    disc_loss = disc_loss_real + disc_loss_fake
    sce_gen = nn.cost.SigmoidCrossEntropy(scope='sce_gen')
    gen_loss = sce_gen(prediction=logit_fake, target=1)

    print discriminator.__dict__
    opt_disc = nn.optimizer.Adam(learning_rate=0.001, scope='TrainDiscriminator/Adam').minimize(
        disc_loss, discriminator.get_parameters_to_train())

    opt_gen = nn.optimizer.Adam(learning_rate=0.001, scope='TrainGenerator/Adam').minimize(
        gen_loss, generator.get_parameters_to_train())

    dataset = _load_data(args.mnist)

if __name__ == '__main__':
    _main()
