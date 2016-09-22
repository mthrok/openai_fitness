import os
import time
import logging

import numpy as np

'''debug
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
'''

import luchador
from luchador.nn import (
    Input,
    Session,
    DeepQLearning,
    SummaryWriter,
    GravesRMSProp,
)
from luchador.nn.util import get_model_config, make_model

logger = logging.getLogger('luchador')

conv_format = luchador.get_nn_conv_format()
min_delta, min_reward = -1.0, -1.0
max_delta, max_reward = 1.0, 1.0

learning_rate = 0.00025
decay1, decay2 = 0.95, 0.95
batchfile = 'mini-batch_Breakout-v0.npz'

n_actions = 6
discount_rate = 0.99
height = width = 84
history = 4
batch = None
state_shape = (
    (batch, height, width, history) if conv_format == 'NHWC' else
    (batch, history, height, width))


def model_maker():
    config = get_model_config('vanilla_dqn', n_actions=n_actions)
    dqn = make_model(config)
    input = Input(shape=state_shape)
    dqn(input())
    return dqn

logger.info('Building Q networks')
ql = DeepQLearning(
    discount_rate=discount_rate,
    min_reward=min_reward,
    max_reward=max_reward,
    min_delta=min_delta,
    max_delta=max_delta,
)
ql.build(model_maker)

logger.info('Building Optimization')
optimizer = GravesRMSProp(
    learning_rate=learning_rate, decay1=decay1, decay2=decay2)
params = ql.pre_trans_net.get_parameter_variables()
minimize_op = optimizer.minimize(ql.error, wrt=params)

logger.info('Initializing Session')
session = Session()
session.initialize()

logger.info('Initializing SummaryWriter')
outputs = ql.pre_trans_net.get_output_tensors()
writer = SummaryWriter('./monitoring/test_tensorflow')
writer.add_graph(session.graph)
writer.register('pre_trans_network_params',
                'histogram', [v.name for v in params])
writer.register('pre_trans_network_outputs',
                'histogram', [v.name for v in outputs])

logger.info('Running computation')
data = np.load(os.path.join(os.path.dirname(__file__), batchfile))
pre_states = data['prestates']
post_states = data['poststates']
actions, rewards = data['actions'], data['rewards']
terminals = data['terminals']
if conv_format == 'NHWC':
    pre_states = pre_states.transpose((0, 2, 3, 1))
    post_states = post_states.transpose((0, 2, 3, 1))

sync_time = []
summary_time = []
run_time = []
filepaths = set()
for i in range(100):
    if i % 10 == 0:
        #######################################################################
        logger.info('Syncing')
        t0 = time.time()
        session.run(name='sync', updates=ql.sync_op)
        sync_time.append(time.time() - t0)

        #######################################################################
        logger.info('Summarizing: {}'.format(i))
        t0 = time.time()
        params_vals = session.run(outputs=params, name='params')
        output_vals = session.run(
            outputs=outputs,
            inputs={ql.pre_states: pre_states},
            name='outputs',
        )
        writer.summarize('pre_trans_network_params', i, params_vals)
        writer.summarize('pre_trans_network_outputs', i, output_vals)
        summary_time.append(time.time() - t0)

    ###########################################################################
    t0 = time.time()
    e, target_qs, future_rewards, post_qs, pre_qs = session.run(
        name='minibatch',
        outputs=[
            ql.error,
            ql.target_q,
            ql.future_reward,
            ql.post_trans_net.output,
            ql.pre_trans_net.output,
        ],
        inputs={
            ql.pre_states: pre_states,
            ql.actions: actions,
            ql.rewards: rewards,
            ql.post_states: post_states,
            ql.terminals: terminals,
        },
        updates=minimize_op)
    run_time.append(time.time() - t0)

logger.info('Sync   : {} ({})'.format(np.mean(sync_time), sync_time[0]))
logger.info('Summary: {} ({})'.format(np.mean(summary_time), summary_time[0]))
logger.info('Run    : {} ({})'.format(np.mean(run_time), run_time[0]))
