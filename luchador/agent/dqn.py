from __future__ import division

import logging
from collections import OrderedDict

import numpy as np

import luchador
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
from luchador.nn import SummaryWriter

from .base import BaseAgent
from .recorder import TransitionRecorder

__all__ = ['DQNAgent']


_LG = logging.getLogger(__name__)


class DQNAgent(BaseAgent):
    def __init__(
            self,
            recorder_config,
            q_network_config,
            optimizer_config,
            action_config,
            training_config,
            save_config,
            summary_config,
    ):
        super(DQNAgent, self).__init__()
        self.recorder_config = recorder_config
        self.q_network_config = q_network_config
        self.optimizer_config = optimizer_config
        self.action_config = action_config
        self.training_config = training_config
        self.save_config = save_config
        self.summary_config = summary_config

    def set_env_info(self, env):
        self._n_actions = env.n_actions

    ###########################################################################
    # Methods for initialization
    def init(self):
        self._init_recorder()
        self._init_counter()
        self._init_network()
        self._init_summary()
        self._init_saver()

    def _init_recorder(self):
        self.recorder = TransitionRecorder(**self.recorder_config)

    def _init_counter(self):
        self.n_total_observations = 0
        self.n_episodes = 0

    def _init_saver(self):
        cfg = self.save_config
        self.saver = Saver(**cfg['saver_config'])

    def _init_summary(self):
        cfg = self.summary_config
        self.summary_writer = SummaryWriter(**cfg['writer_config'])
        self.summary_writer.add_graph(self.session.graph)
        params = self.ql.pre_trans_net.get_parameter_variables()
        outputs = self.ql.pre_trans_net.get_output_tensors()
        self.summary_writer.register(
            'pre_trans_net_params', 'histogram',
            ['/'.join(v.name.split('/')[1:]) for v in params]
        )
        self.summary_writer.register(
            'pre_trans_net_outputs', 'histogram',
            ['/'.join(v.name.split('/')[1:]) for v in outputs]
        )
        self.summary_writer.register(
            'training_summary', 'histogram',
            ['Training/Error', 'Training/Reward', 'Training/Steps']
        )

        self.summary_writer.register(
            'training_summary_ave', 'scalar',
            ['Error/Average', 'Reward/Average', 'Steps/Average']
        )

        self.summary_writer.register(
            'training_summary_min', 'scalar',
            ['Error/Min', 'Reward/Min', 'Steps/Min']
        )

        self.summary_writer.register(
            'training_summary_max', 'scalar',
            ['Error/Max', 'Reward/Max', 'Steps/Max']
        )
        self.summary_values = {
            'error': [],
            'rewards': [],
            'steps': [],
        }

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

    ###########################################################################
    # Methods for `reset`
    def reset(self, initial_observation):
        self.recorder.reset(initial_observation)
        self.n_episodes += 1

    ###########################################################################
    # Methods for `act`
    def act(self):
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
            outputs=self.ql.predicted_q,
            inputs={self.ql.pre_states: state},
            name='action_value',
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
            error = self._train(cfg['n_samples'])
            self.summary_values['error'].append(error)

    def _sync_network(self):
        self.session.run(updates=self.ql.sync_op, name='sync')

    def _train(self, n_samples):
        samples = self.recorder.sample(n_samples)
        error = self.session.run(
            outputs=self.ql.error,
            inputs={
                self.ql.pre_states: samples['pre_states'],
                self.ql.actions: samples['actions'],
                self.ql.rewards: samples['rewards'],
                self.ql.post_states: samples['post_states'],
                self.ql.terminals: samples['terminals'],
            },
            updates=self.minimize_op,
            name='minibatch_training',
        )
        return error

    ###########################################################################
    # Methods for post_episode_action
    def perform_post_episode_task(self, stats):
        self.recorder.truncate()
        self.summary_values['rewards'].append(stats['rewards'])
        self.summary_values['steps'].append(stats['steps'])

        save_interval = self.save_config['interval']
        if save_interval and self.n_episodes % save_interval == 0:
            _LG.info('Saving parameters')
            self.save()

        summary_interval = self.summary_config['interval']
        if summary_interval and self.n_episodes % summary_interval == 0:
            _LG.info('Summarizing Network')
            self.summarize()

    def save(self):
        params = (self.ql.pre_trans_net.get_parameter_variables() +
                  self.optimizer.get_parameter_variables())
        params_val = self.session.run(outputs=params, name='pre_trans_params')
        self.saver.save(OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ]), global_step=self.n_episodes)

    def summarize(self):
        sample = self.recorder.sample(32)

        params = self.ql.pre_trans_net.get_parameter_variables()
        outputs = self.ql.pre_trans_net.get_output_tensors()

        params_vals = self.session.run(outputs=params, name='pre_trans_params')
        output_vals = self.session.run(
            outputs=outputs,
            inputs={self.ql.pre_states: sample['pre_states']},
            name='pre_trans_outputs'
        )
        self.summary_writer.summarize(
            'pre_trans_net_params', self.n_episodes, params_vals)
        self.summary_writer.summarize(
            'pre_trans_net_outputs', self.n_episodes, output_vals)

        summary = self.summary_values
        values = [summary['error'], summary['rewards'], summary['steps']]
        self.summary_writer.summarize(
            'training_summary', self.n_episodes, values,
        )

        self.summary_writer.summarize(
            'training_summary_min', self.n_episodes,
            [np.min(v) if v else 0 for v in values],
        )

        self.summary_writer.summarize(
            'training_summary_ave', self.n_episodes,
            [np.mean(v) if v else 0 for v in values],
        )

        self.summary_writer.summarize(
            'training_summary_max', self.n_episodes,
            [np.max(v) if v else 0 for v in values],
        )

    def __repr__(self):
        ret = '[DQNAgent]\n'
        ret += '  [Recorder]\n'
        for key, value in self.recorder_config.items():
            ret += '    {}: {}\n'.format(key, value)
        ret += '  [Q Network]\n'
        for key, value in self.q_network_config.items():
            ret += '    {}: {}\n'.format(key, value)
        ret += '  [Optimizer]\n'
        for key, value in self.optimizer_config.items():
            ret += '    {}: {}\n'.format(key, value)
        ret += '  [Action]\n'
        for key, value in self.action_config.items():
            ret += '    {}: {}\n'.format(key, value)
        ret += '  [Training]\n'
        for key, value in self.training_config.items():
            ret += '    {}: {}\n'.format(key, value)
        ret += '  [Save]\n'
        for key, value in self.save_config.items():
            ret += '    {}: {}\n'.format(key, value)
        ret += '  [Summary]\n'
        for key, value in self.summary_config.items():
            ret += '    {}: {}\n'.format(key, value)
        return ret
