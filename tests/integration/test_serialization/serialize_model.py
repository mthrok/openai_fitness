"""Build network model and run optimization, then saven variables"""
import logging
from collections import OrderedDict

import numpy as np

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'

import luchador
from luchador.util import load_config
from luchador import nn

_LG = logging.getLogger('luchador')
logging.getLogger('luchador.nn.saver').setLevel(logging.DEBUG)

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
    return nn.get_optimizer(cfg['name'])(**cfg['args'])


def _build_network(model_filepath, optimizer_filepath):
    max_reward, min_reward = 1.0, -1.0
    max_delta, min_delta = 1.0, -1.0

    discount_rate = 0.99

    def _model_maker():
        config = nn.get_model_config(model_filepath, n_actions=N_ACTIONS)
        dqn = nn.make_model(config)
        input_tensor = nn.Input(shape=SHAPE, name='input')
        dqn(input_tensor)
        return dqn

    _LG.info('Building Q networks')
    ql = luchador.nn.q_learning.DeepQLearning(
        discount_rate=discount_rate,
        min_reward=min_reward, max_reward=max_reward,
        min_delta=min_delta, max_delta=max_delta)
    ql.build(_model_maker)

    _LG.info('Building Optimization')
    optimizer = _make_optimizer(optimizer_filepath)
    minimize_op = optimizer.minimize(
        ql.error, wrt=ql.pre_trans_net.get_parameter_variables())

    return ql, optimizer, minimize_op


def _serialize(filepath, components, session):
    saver = luchador.nn.Saver(filepath)
    _LG.info('Save parameters to %s', filepath)
    params = []
    for component in components:
        params.extend(component.get_parameter_variables())
    if params:
        params_val = session.run(outputs=params)
        saver.save(OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ]), global_step=0)


def _deserialize(filepath, components, session):
    _LG.info('Loading parameters from %s', filepath)
    var_names = [
        var.name for component in components
        for var in component.get_parameter_variables()
    ]
    session.load_from_file(filepath, var_names)


def _run(session, qln, minimize_op):
    _LG.info('Running minimization op')

    session.run(updates=qln.sync_op)
    pre_states = np.random.randint(
        low=0, high=256, size=SHAPE, dtype=np.uint8)
    post_states = np.random.randint(
        low=0, high=256, size=SHAPE, dtype=np.uint8)
    actions = np.random.randint(
        low=0, high=N_ACTIONS, size=(BATCH_SIZE,), dtype=np.uint8)
    rewards = np.random.randint(
        low=0, high=2, size=(BATCH_SIZE,), dtype=np.uint8)
    terminals = np.random.randint(
        low=0, high=2, size=(BATCH_SIZE,), dtype=np.bool)

    session.run(
        name='minibatch',
        inputs={
            qln.pre_states: pre_states,
            qln.actions: actions,
            qln.rewards: rewards,
            qln.post_states: post_states,
            qln.terminals: terminals,
        },
        updates=minimize_op
    )


def _main():
    args = _parse_command_line_args()
    session = nn.Session()
    ql, optimizer, minimize_op = _build_network(
        model_filepath=args.model,
        optimizer_filepath=args.optimizer,
    )

    components = [ql.pre_trans_net, optimizer]
    if args.input:
        _deserialize(args.input, components, session)
    else:
        _LG.info('Initializing Session')
        session.initialize()

    _run(session, ql, minimize_op)

    if args.output:
        _serialize(args.output, components, session)


if __name__ == '__main__':
    _main()
