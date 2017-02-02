"""Vanilla DQNAgent from [1]_:

References
----------
.. [1] Mnih, V et. al (2015)
       Human-level control through deep reinforcement learning
       https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""
from __future__ import division

import logging

import numpy as np

import luchador
import luchador.util

from .base import BaseAgent
from .recorder import TransitionRecorder
from .misc import EGreedy
from .rl import DeepQLearning

__all__ = ['DQNAgent']


_LG = logging.getLogger(__name__)


def _transpose(state):
    return state.transpose((0, 2, 3, 1))


class DQNAgent(luchador.util.StoreMixin, BaseAgent):
    """Implement Vanilla DQNAgent from [1]_:

    References
    ----------
    .. [1] Mnih, V et. al (2015)
        Human-level control through deep reinforcement learning
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    def __init__(
            self,
            recorder_config,
            q_network_config,
            save_config,
            summary_config,
            action_config,
            training_config,
    ):
        super(DQNAgent, self).__init__()
        self._store_args(
            recorder_config=recorder_config,
            q_network_config=q_network_config,
            save_config=save_config,
            summary_config=summary_config,
            action_config=action_config,
            training_config=training_config,
        )
        self._n_obs = 0
        self._n_train = 0
        self._n_actions = None

        self._recorder = None
        self._ql = None
        self._eg = None
        self._summary_values = {
            'errors': [],
            'rewards': [],
            'steps': [],
            'episode': 0,
        }

    ###########################################################################
    # Methods for initialization
    def init(self, env):
        self._n_actions = env.n_actions
        self._recorder = TransitionRecorder(**self.args['recorder_config'])

        self._init_network(n_actions=env.n_actions)
        self._eg = EGreedy(**self.args['action_config'])

    def _init_network(self, n_actions):
        cfg = self.args['q_network_config']
        self._ql = DeepQLearning(
            model_config=cfg['model_config'],
            q_learning_config=cfg['q_learning_config'],
            cost_config=cfg['cost_config'],
            optimizer_config=cfg['optimizer_config'],
            summary_writer_config=cfg['summary_writer_config'],
            saver_config=cfg['saver_config'],
        )
        self._ql.build(n_actions=n_actions)
        self._ql.sync_network()
        self._ql.summarize_layer_params()

    ###########################################################################
    # Methods for `reset`
    def reset(self, initial_observation):
        self._recorder.reset(
            initial_data={'state': initial_observation})

    ###########################################################################
    # Methods for `act`
    def act(self):
        if (
                not self._recorder.is_ready() or
                self._eg.act_random()
        ):
            return np.random.randint(self._n_actions)

        q_val = self._predict_q()
        return np.argmax(q_val)

    def _predict_q(self):
        # _LG.debug('Predicting Q value from NN')
        state = self._recorder.get_last_stack()['state'][None, ...]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        return self._ql.predict_action_value(state)[0]

    ###########################################################################
    # Methods for `learn`
    def learn(self, state0, action, reward, state1, terminal, info=None):
        self._recorder.record({
            'action': action, 'reward': reward,
            'state': state1, 'terminal': terminal})
        self._n_obs += 1

        cfg, n_obs = self.args['training_config'], self._n_obs
        if cfg['train_start'] < 0 or n_obs < cfg['train_start']:
            return

        if n_obs == cfg['train_start']:
            _LG.info('Starting DQN training')

        if n_obs % cfg['sync_frequency'] == 0:
            self._ql.sync_network()

        if n_obs % cfg['train_frequency'] == 0:
            error = self._train(cfg['n_samples'])
            self._n_train += 1
            self._summary_values['errors'].append(error)

            interval = self.args['save_config']['interval']
            if interval > 0 and self._n_train % interval == 0:
                _LG.info('Saving parameters')
                self._ql.save()

            interval = self.args['summary_config']['interval']
            if interval > 0 and self._n_train % interval == 0:
                _LG.info('Summarizing Network')
                self._ql.summarize_layer_params()
                self._summarize_layer_outputs()
                self._summarize_history()

    def _train(self, n_samples):
        samples = self._recorder.sample(n_samples)
        state0 = samples['state'][0]
        state1 = samples['state'][1]
        if luchador.get_nn_conv_format() == 'NHWC':
            state0 = _transpose(state0)
            state1 = _transpose(state1)
        return self._ql.train(
            state0, samples['action'], samples['reward'],
            state1, samples['terminal'])

    def _summarize_layer_outputs(self):
        sample = self._recorder.sample(32)
        state = sample['state'][0]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        self._ql.summarize_layer_outputs(state)

    def _summarize_history(self):
        self._ql.summarize_stats(**self._summary_values)
        self._summary_values['errors'] = []
        self._summary_values['rewards'] = []
        self._summary_values['steps'] = []

    ###########################################################################
    # Methods for post_episode_action
    def perform_post_episode_task(self, stats):
        self._recorder.truncate()
        self._summary_values['rewards'].append(stats['rewards'])
        self._summary_values['steps'].append(stats['steps'])
        self._summary_values['episode'] = stats['episode']
