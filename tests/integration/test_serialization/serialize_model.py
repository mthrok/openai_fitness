"""Build network model and run optimization, then saven variables"""
import logging

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador.util import load_config, initialize_logger
from luchador import nn
from luchador.nn.saver import Saver
from luchador.agent.rl.q_learning import DeepQLearning

_LG = logging.getLogger('luchador')

WIDTH = 84
HEIGHT = 84
CHANNEL = 4
BATCH_SIZE = 32
N_ACTIONS = 6
SHAPE = (
    (BATCH_SIZE, HEIGHT, WIDTH, CHANNEL)
    if luchador.get_nn_conv_format() == 'NHWC' else
    (BATCH_SIZE, CHANNEL, HEIGHT, WIDTH)
)


def _parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description=(
            'Build Network model and optimization, '
            'and serialize variables with Saver'
        )
    )
    ap.add_argument(
        'model',
        help='Model definition YAML file. '
    )
    ap.add_argument(
        'optimizer',
        help='Optimizer configuration YAML file.'
    )
    ap.add_argument(
        '--output',
        help='File path to save parameters'
    )
    ap.add_argument(
        '--input',
        help='Path to parameter file from which data is loaded'
    )
    return ap.parse_args()


def _make_optimizer(filepath):
    cfg = load_config(filepath)
    return nn.get_optimizer(cfg['typename'])(**cfg['args'])


def _gen_model_def(model_file):
    fmt = luchador.get_nn_conv_format()
    w, h, c = WIDTH, HEIGHT, CHANNEL
    shape = (
        '[null, {}, {}, {}]'.format(h, w, c) if fmt == 'NHWC' else
        '[null, {}, {}, {}]'.format(c, h, w)
    )
    return nn.get_model_config(
        model_file, n_actions=N_ACTIONS, input_shape=shape)


def _build_network(model_filepath, optimizer_filepath, initial_parameter):
    _LG.info('Building Q networks')
    dql = DeepQLearning(
        q_learning_config={
            'discount_rate': 0.99,
            'min_reward': -1.0,
            'max_reward': 1.0,
        },
        optimizer_config=load_config(optimizer_filepath),
    )
    model_def = _gen_model_def(model_filepath)
    dql.build(model_def, initial_parameter)
    _LG.info('Syncing models')
    dql.sync_network()
    return dql


def _run(dql):
    state_0 = np.random.randint(
        low=0, high=256, size=SHAPE, dtype=np.uint8)
    state_1 = np.random.randint(
        low=0, high=256, size=SHAPE, dtype=np.uint8)
    action = np.random.randint(
        low=0, high=N_ACTIONS, size=(BATCH_SIZE,), dtype=np.uint8)
    reward = np.random.randint(
        low=0, high=2, size=(BATCH_SIZE,), dtype=np.uint8)
    terminal = np.random.randint(
        low=0, high=2, size=(BATCH_SIZE,), dtype=np.bool)
    _LG.info('Running minimization op')
    dql.train(state_0, action, reward, state_1, terminal)


def _initialize_logger():
    initialize_logger(
        name='luchador', level=logging.INFO,
        message_format='%(asctime)s: %(levelname)5s: %(message)s'
    )
    logging.getLogger('luchador.nn.saver').setLevel(logging.DEBUG)


def _main():
    args = _parse_command_line_args()
    _initialize_logger()

    dql = _build_network(
        model_filepath=args.model,
        optimizer_filepath=args.optimizer,
        initial_parameter=args.input,
    )

    _run(dql)

    if args.output:
        saver = Saver(output_dir=args.output)
        saver.save(dql.fetch_all_parameters(), global_step=1)


if __name__ == '__main__':
    _main()
