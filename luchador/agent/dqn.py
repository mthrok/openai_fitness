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
from luchador import nn

from .base import BaseAgent
from .recorder import TransitionRecorder
from .misc import EGreedy
from .rl import DeepQLearning

__all__ = ['DQNAgent']


_LG = logging.getLogger(__name__)


def _transpose(state):
    return state.transpose((0, 2, 3, 1))


class DQNAgent(BaseAgent):
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
        self.recorder_config = recorder_config
        self.q_network_config = q_network_config
        self.save_config = save_config
        self.summary_config = summary_config
        self.action_config = action_config
        self.training_config = training_config

        self.n_observations = 0
        self.n_trainings = 0

        self._n_actions = None
        self.recorder = None
        self.ql = None
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
        self.recorder = TransitionRecorder(**self.recorder_config)

        self._init_network()

        self.ql.summarize_layer_params()
        self._eg = EGreedy(
            epsilon_init=self.action_config['initial_exploration_rate'],
            epsilon_term=self.action_config['terminal_exploration_rate'],
            duration=self.action_config['exploration_period'],
            method=self.action_config['annealing_method'],
        )

    def _init_network(self):
        self._build_network()
        self.ql.sync_network()

    def _build_network(self):
        cfg = self.q_network_config
        w, h, c = cfg['state_width'], cfg['state_height'], cfg['state_length']
        model_name = cfg['model_name']

        fmt = luchador.get_nn_conv_format()
        shape = (None, h, w, c) if fmt == 'NHWC' else (None, c, h, w)

        model_def = nn.get_model_config(model_name, n_actions=self._n_actions)

        def _model_maker():
            dqn = nn.make_model(model_def)
            input_tensor = nn.Input(shape=shape)
            dqn(input_tensor)
            return dqn

        self.ql = DeepQLearning(
            q_learning_config=cfg['q_learning_config'],
            optimizer_config=cfg['optimizer_config'],
            summary_writer_config=cfg['summary_writer_config'],
            saver_config=cfg['saver_config'],
            session_config=cfg['session_config']
        )
        self.ql.build(_model_maker)

    ###########################################################################
    # Methods for `reset`
    def reset(self, initial_observation):
        self.recorder.reset(
            initial_data={'state': initial_observation})

    ###########################################################################
    # Methods for `act`
    def act(self):
        if (
                not self.recorder.is_ready() or
                self._eg.act_random()
        ):
            return np.random.randint(self._n_actions)

        q_val = self._predict_q()
        return np.argmax(q_val)

    def _predict_q(self):
        # _LG.debug('Predicting Q value from NN')
        state = self.recorder.get_last_stack()['state'][None, ...]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        return self.ql.predict_action_value(state)[0]

    ###########################################################################
    # Methods for `learn`
    def learn(self, state0, action, reward, state1, terminal, info=None):
        self.recorder.record({
            'action': action, 'reward': reward,
            'state': state1, 'terminal': terminal})
        self.n_observations += 1

        cfg, n_obs = self.training_config, self.n_observations
        if cfg['train_start'] < 0 or n_obs < cfg['train_start']:
            return

        if n_obs == cfg['train_start']:
            _LG.info('Starting DQN training')

        if n_obs % cfg['sync_frequency'] == 0:
            self.ql.sync_network()

        if n_obs % cfg['train_frequency'] == 0:
            error = self._train(cfg['n_samples'])
            self.n_trainings += 1
            self._summary_values['errors'].append(error)

            interval = self.save_config['interval']
            if interval > 0 and self.n_trainings % interval == 0:
                _LG.info('Saving parameters')
                self.ql.save()

            interval = self.summary_config['interval']
            if interval > 0 and self.n_trainings % interval == 0:
                _LG.info('Summarizing Network')
                self.ql.summarize_layer_params()
                self._summarize_layer_outputs()
                self._summarize_history()

    def _train(self, n_samples):
        samples = self.recorder.sample(n_samples)
        state0 = samples['state'][0]
        state1 = samples['state'][1]
        if luchador.get_nn_conv_format() == 'NHWC':
            state0 = _transpose(state0)
            state1 = _transpose(state1)
        return self.ql.train(
            state0, samples['action'], samples['reward'],
            state1, samples['terminal'])

    def _summarize_layer_outputs(self):
        sample = self.recorder.sample(32)
        state = sample['state'][0]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        self.ql.summarize_layer_outputs(state)

    def _summarize_history(self):
        self.ql.summarize_stats(**self._summary_values)
        self._summary_values['errors'] = []
        self._summary_values['rewards'] = []
        self._summary_values['steps'] = []

    ###########################################################################
    # Methods for post_episode_action
    def perform_post_episode_task(self, stats):
        self.recorder.truncate()
        self._summary_values['rewards'].append(stats['rewards'])
        self._summary_values['steps'].append(stats['steps'])
        self._summary_values['episode'] = stats['episode']

    ###########################################################################
    def __repr__(self):
        return luchador.util.pprint_dict({
            self.__class__.__name__: {
                'Recorder': self.recorder_config,
                'Q Network': self.q_network_config,
                'Action': self.action_config,
                'Training': self.training_config,
                'Save': self.save_config,
                'Summary': self.summary_config,
                }
        })
