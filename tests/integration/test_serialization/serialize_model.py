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


def parse_command_line_args():
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


def make_optimizer(filepath):
    cfg = load_config(filepath)
    return nn.get_optimizer(cfg['name'])(**cfg['args'])


def build_network(model_filepath, optimizer_filepath):
    max_reward, min_reward = 1.0, -1.0
    max_delta, min_delta = 1.0, -1.0

    discount_rate = 0.99

    def model_maker():
        config = nn.get_model_config(model_filepath, n_actions=N_ACTIONS)
        dqn = nn.make_model(config)
        input_tensor = nn.Input(shape=SHAPE, name='input')
        dqn(input_tensor())
        return dqn

    _LG.info('Building Q networks')
    ql = luchador.nn.q_learning.DeepQLearning(
        discount_rate=discount_rate,
        min_reward=min_reward, max_reward=max_reward,
        min_delta=min_delta, max_delta=max_delta)
    ql.build(model_maker)

    _LG.info('Building Optimization')
    optimizer = make_optimizer(optimizer_filepath)
    minimize_op = optimizer.minimize(
        ql.error, wrt=ql.pre_trans_net.get_parameter_variables())

    return ql, optimizer, minimize_op


def serialize(filepath, components, session):
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


def deserialize(filepath, components, session):
    _LG.info('Loading parameters from %s', filepath)
    var_names = [
        var.name for component in components
        for var in component.get_parameter_variables()
    ]
    session.load_from_file(filepath, var_names)


def run(session, ql, minimize_op):
    _LG.info('Running minimization op')

    session.run(updates=ql.sync_op)
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
            ql.pre_states: pre_states,
            ql.actions: actions,
            ql.rewards: rewards,
            ql.post_states: post_states,
            ql.terminals: terminals,
        },
        updates=minimize_op
    )


def main():
    args = parse_command_line_args()
    session = nn.Session()
    ql, optimizer, minimize_op = build_network(
        model_filepath=args.model,
        optimizer_filepath=args.optimizer,
    )

    components = [ql.pre_trans_net, optimizer]
    if args.input:
        deserialize(args.input, components, session)
    else:
        _LG.info('Initializing Session')
        session.initialize()

    run(session, ql, minimize_op)

    if args.output:
        serialize(args.output, components, session)


if __name__ == '__main__':
    main()
