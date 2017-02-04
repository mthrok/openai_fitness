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


class DQNAgent(luchador.util.StoreMixin, BaseAgent):
    """Implement Vanilla DQNAgent from [1]_:

    Parameters
    ----------
    recorder_config : dict
        Constructor arguments for
        :class:`luchador.agent.recorder.TransitionRecorder`

    model_config : dict
        Configuration for model definition.

        name : str
            The name of network model or path to model definition file.
        initial_parameter : str
            Path to HDF5 file which contain the initial parameter values.
        input_channel, input_height, input_width : int
            The shape of input to the network

    q_network_config : dict
        Constructor arguments for
        :class:`luchador.agent.rl.q_learning.DeepQLearning`

    saver_config : dict
        Constructor arguments for :class:`luchador.nn.saver.Saver`

    summary_writer_config : dict
        Constructor arguments for :class:`luchador.nn.summary.SummaryWriter`

    action_config : dict
        Constructor arguments for :class:`luchador.agent.misc.EGreedy`

    training_config : dict
        Configuration for training

        train_start : int
            Training starts after this number of transitions are recorded
            Giving negative value effectively disable training and network sync
        train_frequency : int
            Train network every this number of observations are made
        sync_frequency : int
            Sync networks every this number of observations are made
        n_samples : int
            Batch size

    References
    ----------
    .. [1] Mnih, V et. al (2015)
        Human-level control through deep reinforcement learning
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    def __init__(
            self,
            recorder_config,
            model_config,
            q_network_config,
            action_config,
            training_config,
            saver_config,
            save_config,
            summary_writer_config,
            summary_config,
    ):
        super(DQNAgent, self).__init__()
        self._store_args(
            recorder_config=recorder_config,
            model_config=model_config,
            q_network_config=q_network_config,
            action_config=action_config,
            training_config=training_config,
            saver_config=saver_config,
            save_config=save_config,
            summary_writer_config=summary_writer_config,
            summary_config=summary_config,
        )
        self._n_obs = 0
        self._n_train = 0
        self._n_actions = None

        self._recorder = None
        self._saver = None
        self._ql = None
        self._eg = None
        self._summary_writer = None
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
        self._init_saver()
        self._init_summary_writer()
        self._summarize_layer_params()

    def _gen_model_def(self, n_actions):
        cfg = self.args['model_config']
        fmt = luchador.get_nn_conv_format()
        w, h, c = cfg['input_width'], cfg['input_height'], cfg['input_channel']
        shape = (
            '[null, {}, {}, {}]'.format(h, w, c) if fmt == 'NHWC' else
            '[null, {}, {}, {}]'.format(c, h, w)
        )
        return nn.get_model_config(
            cfg['model_file'], n_actions=n_actions, input_shape=shape)

    def _init_network(self, n_actions):
        cfg = self.args['q_network_config']
        self._ql = DeepQLearning(
            q_learning_config=cfg['q_learning_config'],
            cost_config=cfg['cost_config'],
            optimizer_config=cfg['optimizer_config'],
        )

        model_def = self._gen_model_def(n_actions)
        initial_parameter = self.args['model_config']['initial_parameter']
        self._ql.build(model_def, initial_parameter)
        self._ql.sync_network()

    def _init_saver(self):
        config = self.args['saver_config']
        self._saver = nn.Saver(**config)

    def _init_summary_writer(self):
        """Initialize SummaryWriter and create set of summary operations"""
        config = self.args['summary_writer_config']
        self._summary_writer = nn.SummaryWriter(**config)

        if self._ql.session.graph:
            self._summary_writer.add_graph(self._ql.session.graph)

        model_0 = self._ql.models['model_0']
        params = model_0.get_parameter_variables()
        outputs = model_0.get_output_tensors()
        self._summary_writer.register(
            'histogram', tag='params',
            names=['/'.join(v.name.split('/')[1:]) for v in params])
        self._summary_writer.register(
            'histogram', tag='outputs',
            names=['/'.join(v.name.split('/')[1:]) for v in outputs])
        self._summary_writer.register(
            'histogram', tag='training',
            names=['Training/Error', 'Training/Reward', 'Training/Steps']
        )
        self._summary_writer.register_stats(['Error', 'Reward', 'Steps'])
        self._summary_writer.register('scalar', ['Trainings'])

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
                self._save()

            interval = self.args['summary_config']['interval']
            if interval > 0 and self._n_train % interval == 0:
                _LG.info('Summarizing Network')
                self._summarize_layer_params()
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

    def _save(self):
        data = self._ql.fetch_all_parameters()
        self._saver.save(data, global_step=self._n_train)

    def _summarize_layer_params(self):
        dataset = self._ql.fetch_layer_params()
        self._summary_writer.summarize(
            global_step=self._n_train, dataset=dataset)

    def _summarize_layer_outputs(self):
        sample = self._recorder.sample(32)
        state = sample['state'][0]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        dataset = self._ql.fetch_layer_outputs(state)
        self._summary_writer.summarize(
            global_step=self._n_train, dataset=dataset)

    def _summarize_history(self):
        steps = self._summary_values['steps']
        errors = self._summary_values['errors']
        rewards = self._summary_values['rewards']
        episode = self._summary_values['episode']
        self._summary_writer.summarize(
            global_step=self._n_train, tag='training',
            dataset=[errors, rewards, steps],
        )
        self._summary_writer.summarize(
            global_step=episode, dataset={'Trainings': self._n_train}
        )
        if rewards:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'Reward': rewards}
            )
        if errors:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'Error': errors}
            )
        if steps:
            self._summary_writer.summarize_stats(
                global_step=self._n_train, dataset={'Steps': steps}
            )
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
