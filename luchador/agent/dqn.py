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

    ###########################################################################
    # Methods for initialization
    def init(self, env):
        self._n_actions = env.n_actions
        self.n_total_observations = 0
        self.recorder = TransitionRecorder(**self.recorder_config)

        self._init_network()
        self._init_summary_writer()
        self.saver = Saver(**self.save_config['saver_config'])

    def _init_summary_writer(self):
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
        model_name = cfg['model_name']

        fmt = luchador.get_nn_conv_format()
        shape = (None, h, w, c) if fmt == 'NHWC' else (None, c, h, w)

        model_def = get_model_config(model_name, n_actions=self._n_actions)

        def model_maker():
            dqn = make_model(model_def)
            input = Input(shape=shape)
            dqn(input())
            return dqn

        self.ql = DeepQLearning(**cfg['args'])
        self.ql.build(model_maker)

    def _build_optimization(self):
        self.optimizer = make_optimizer(self.optimizer_config)
        wrt = self.ql.pre_trans_net.get_parameter_variables()
        self.minimize_op = self.optimizer.minimize(self.ql.error, wrt=wrt)

    def _init_session(self):
        cfg = self.q_network_config
        self.session = Session()
        if cfg.get('parameter_file'):
            _LG.info('Loading paramter from {}'.format(cfg['parameter_file']))
            self.session.load_from_file(cfg['parameter_file'])
        else:
            self.session.initialize()

    ###########################################################################
    # Methods for `reset`
    def reset(self, initial_observation):
        self.recorder.reset(initial_observation)

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
    def observe(self, action, outcome):
        self.recorder.record(
            action=action, reward=outcome.reward,
            observation=outcome.observation, terminal=outcome.terminal)
        self.n_total_observations += 1

        cfg, n_obs = self.training_config, self.n_total_observations
        if cfg['train_start'] < 0 or n_obs < cfg['train_start']:
            return

        if n_obs == cfg['train_start']:
            _LG.info('Starting DQN training')

        if n_obs % cfg['sync_frequency'] == 0:
            self._sync_network()

        if n_obs % cfg['train_frequency'] == 0:
            error = self._train(cfg['n_samples'])
            self.summary_values['error'].append(error)

    def _sync_network(self):
        self.session.run(updates=self.ql.sync_op, name='sync')

    def _train(self, n_samples):
        samples = self.recorder.sample(n_samples)
        updates = self.ql.pre_trans_net.get_update_operations() + [
            self.minimize_op]
        error = self.session.run(
            outputs=self.ql.error,
            inputs={
                self.ql.pre_states: samples['pre_states'],
                self.ql.actions: samples['actions'],
                self.ql.rewards: samples['rewards'],
                self.ql.post_states: samples['post_states'],
                self.ql.terminals: samples['terminals'],
            },
            updates=updates,
            name='minibatch_training',
        )
        return error

    ###########################################################################
    # Methods for post_episode_action
    def perform_post_episode_task(self, stats):
        self.recorder.truncate()
        episode = stats['episode']
        self.summary_values['rewards'].append(stats['rewards'])
        self.summary_values['steps'].append(stats['steps'])

        save_interval = self.save_config['interval']
        if save_interval > 0 and episode % save_interval == 0:
            _LG.info('Saving parameters')
            self._save(episode)

        summary_interval = self.summary_config['interval']
        if summary_interval > 0 and episode % summary_interval == 0:
            _LG.info('Summarizing Network')
            self._summarize(episode)

    def _save(self, episode):
        """Save network parameter to file"""
        params = (self.ql.pre_trans_net.get_parameter_variables() +
                  self.optimizer.get_parameter_variables())
        params_val = self.session.run(outputs=params, name='pre_trans_params')
        self.saver.save(OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ]), global_step=episode)

    def _summarize(self, episode):
        """Summarize network parameter, output and training history"""
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
            'pre_trans_net_params', episode, params_vals)
        self.summary_writer.summarize(
            'pre_trans_net_outputs', episode, output_vals)

        summary = self.summary_values
        values = [summary['error'], summary['rewards'], summary['steps']]
        self.summary_writer.summarize(
            'training_summary', episode, values,
        )

        self.summary_writer.summarize(
            'training_summary_min', episode,
            [np.min(v) if v else 0 for v in values],
        )

        self.summary_writer.summarize(
            'training_summary_ave', episode,
            [np.mean(v) if v else 0 for v in values],
        )

        self.summary_writer.summarize(
            'training_summary_max', episode,
            [np.max(v) if v else 0 for v in values],
        )
        self.summary_values = {
            'error': [],
            'rewards': [],
            'steps': [],
        }

    ###########################################################################
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
