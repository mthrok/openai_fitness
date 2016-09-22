from __future__ import division

import os
import logging
import warnings

import numpy as np

import luchador
from luchador.util import load_config
from luchador.nn import (
    Session,
    Input,
    Saver,
    DeepQLearning,
    make_optimizer,
)
from luchador.nn.util import (
    make_model,
    get_model_config,
)
# from luchador.nn import SummaryWriter

from .base import BaseAgent
from .recorder import TransitionRecorder

__all__ = ['DQNAgent']


_LG = logging.getLogger(__name__)
_DEFAULT_CONFIG = load_config(
    os.path.join(os.path.dirname(__file__), 'vanilla_dqn.yml'))


class DQNAgent(BaseAgent):
    def __init__(
            self,
            recorder_config=_DEFAULT_CONFIG['recorder_config'],
            q_network_config=_DEFAULT_CONFIG['q_network_config'],
            optimizer_config=_DEFAULT_CONFIG['optimizer_config'],
            action_config=_DEFAULT_CONFIG['action_config'],
            training_config=_DEFAULT_CONFIG['training_config'],
            save_config=_DEFAULT_CONFIG['save_config'],
    ):
        super(DQNAgent, self).__init__()
        self.recorder_config = recorder_config
        self.q_network_config = q_network_config
        self.optimizer_config = optimizer_config
        self.action_config = action_config
        self.training_config = training_config
        self.save_config = save_config

    def set_env_info(self, env):
        self._n_actions = env.n_actions

    ###########################################################################
    # Methods for initialization
    def init(self):
        self._init_recorder()
        self._init_counter()
        self._init_network()
        self._init_saver()
        warnings.warn('Not completed yet.')

    def _init_recorder(self):
        self.recorder = TransitionRecorder(**self.recorder_config)

    def _init_counter(self):
        self.n_total_observations = 0
        self.n_episodes = 0

    def _init_network(self):
        self._build_network()
        self._build_optimization()
        self._init_session()
        self._sync_network()

    def _build_network(self):
        cfg = self.q_network_config
        w, h, c = cfg['state_width'], cfg['state_height'], cfg['state_length']
        network_name = cfg['network_name']

        fmt = luchador.get_nn_conv_format()
        shape = (None, h, w, c) if fmt == 'NHWC' else (None, c, h, w)

        model_def = get_model_config(network_name, n_actions=self._n_actions)

        def model_maker():
            dqn = make_model(model_def)
            input = Input(shape=shape)
            dqn(input())
            return dqn

        self.ql = DeepQLearning(discount_rate=cfg['discount_rate'],
                                min_delta=cfg['min_delta'],
                                max_delta=cfg['max_delta'],
                                min_reward=cfg['min_reward'],
                                max_reward=cfg['max_reward'])
        self.ql.build(model_maker)

    def _build_optimization(self):
        self.optimizer = make_optimizer(self.optimizer_config)
        wrt = self.ql.pre_trans_net.get_parameter_variables()
        self.minimize_op = self.optimizer.minimize(self.ql.error, wrt=wrt)

    def _init_session(self):
        self.session = Session()
        self.session.initialize()

    def _init_saver(self):
        cfg = self.save_config
        self.saver = Saver(cfg['output_dir'])

    ###########################################################################
    # Methods for `reset`
    def reset(self, initial_observation):
        warnings.warn('Not completed yet.')
        self.recorder.reset(initial_observation)
        self.n_episodes += 1

    ###########################################################################
    # Methods for `act`
    def act(self):
        warnings.warn('Not completed yet.')
        if (
                not self.recorder.is_ready() or
                np.random.rand() < self._get_exploration_probability()
        ):
            return np.random.randint(self._n_actions)

        q_val = self._predict_q()
        return np.argmax(q_val)

    def _get_exploration_probability(self):
        r_init = self.action_config['initial_exploration_rate']
        r_term = self.action_config['terminal_exploration_rate']
        t_end = self.action_config['exploration_period']
        t_now = self.n_total_observations
        if t_now < t_end:
            return r_init - t_now * (r_init - r_term) / t_end
        return r_term

    def _predict_q(self):
        # _LG.debug('Predicting Q value from NN')
        state = self.recorder.get_current_state()
        q_val = self.session.run(
            name='action_value',
            outputs=self.ql.predicted_q,
            inputs={self.ql.pre_states: state},
        )
        return q_val[0]

    ###########################################################################
    # Methods for `observe`
    def observe(self, action, observation, reward, terminal, env_state):
        self.recorder.record(action=action, reward=reward,
                             observation=observation, terminal=terminal)
        self.n_total_observations += 1

        cfg = self.training_config
        train_start = cfg['train_start']
        train_freq, sync_freq = cfg['train_frequency'], cfg['sync_frequency']
        n_obs = self.n_total_observations

        if sync_freq and n_obs % sync_freq == 1:
            self._sync_network()

        if n_obs > train_start and n_obs % train_freq == 0:
            self._train(cfg['n_samples'])

        warnings.warn('Not completed yet.')

    def _sync_networkg(self):
        self.session.run(name='sync', updates=self.ql.sync_op)

    def _train(self, n_samples):
        samples = self.recorder.sample(n_samples)
        error = self.session.run(
            name='minibatch',
            outputs=self.ql.error,
            inputs={
                self.ql.pre_states: samples['pre_states'],
                self.ql.actions: samples['actions'],
                self.ql.rewards: samples['rewards'],
                self.ql.post_states: samples['post_states'],
                self.ql.terminals: samples['terminals'],
            },
            updates=self.minimize_op
        )
        return error

    ###########################################################################
    # Methods for post_episode_action
    def perform_post_episode_task(self):
        self.recorder.truncate()
        save_interval = self.save_config['interval']
        if save_interval and self.n_episodes % save_interval == 0:
            self.save()

    def save(self):
        raise NotImplementedError('`save` is not implemented')
