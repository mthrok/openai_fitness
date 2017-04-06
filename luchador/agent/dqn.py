"""Implement Agent part of Deep Q Learning"""
from __future__ import division

import os.path
import logging

import numpy as np

import luchador
import luchador.util
from luchador import nn

from .base import BaseAgent
from .recorder import PrioritizedQueue
from .misc import EGreedy
from .rl import DeepQLearning, DoubleDeepQLearning

__all__ = ['DQNAgent']


_LG = logging.getLogger(__name__)


def _transpose(state):
    return state.transpose((0, 2, 3, 1))


def _initialize_summary_writer(config):
    writer = nn.SummaryWriter(**config)
    sess = nn.get_session()
    if sess.graph:
        writer.add_graph(sess.graph)
    return writer


def _gen_model_def(config, n_actions):
    fmt, w = luchador.get_nn_conv_format(), config['input_width']
    h, c = config['input_height'], config['input_channel']
    model = config['model_file']
    shape = [None, h, w, c] if fmt == 'NHWC' else [None, c, h, w]
    return nn.get_model_config(model, n_actions=n_actions, input_shape=shape)


def _get_q_network(config):
    if config['typename'] == 'DeepQLearning':
        dqn = DeepQLearning
    elif config['typename'] == 'DoubleDeepQLearning':
        dqn = DoubleDeepQLearning
    return dqn(**config['args'])


class DQNAgent(luchador.util.StoreMixin, BaseAgent):  # pylint: disable=R0902
    """Implement Agent part of DQN [1]_:

    Parameters
    ----------
    record_config : dict
        Configuration for recording

        stack : int
            #states to stack
        sort_frequency : int
            Sort heap buffer every this number of records are put.
            Giving non-positive value will disable this.
        save_frequency : int
            Save heap buffer every this number of records are put.
            This functionality is for inspecting the replay memory, and not
            meant for deserializing replay memory.
            Giving non-positive value will disable this.
        save_dir : str
            Save directory name.

    recorder_config : dict
        Constructor arguments for
        :class:`luchador.agent.recorder.PrioritizedQueue`

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
            record_config,
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
            record_config=record_config,
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
        self._ready = False

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
        self._recorder = PrioritizedQueue(**self.args['recorder_config'])

        self._init_network(n_actions=env.n_actions)
        self._eg = EGreedy(**self.args['action_config'])
        self._saver = nn.Saver(**self.args['saver_config'])
        self._summary_writer = _initialize_summary_writer(
            config=self.args['summary_writer_config'],
        )
        self._summarize_layer_params()

    def _init_network(self, n_actions):
        model_def = _gen_model_def(self.args['model_config'], n_actions)
        initial_parameter = self.args['model_config']['initial_parameter']
        _LG.info('\n%s', luchador.util.pprint_dict(model_def))
        self._ql = _get_q_network(config=self.args['q_network_config'])
        self._ql.build(model_def)
        self._ql.initialize(initial_parameter)
        self._ql.sync_network()

    ###########################################################################
    # Methods for `reset`
    def reset(self, _):
        self._ready = False

    ###########################################################################
    # Methods for `act`
    def act(self):
        if not self._ready or self._eg.act_random():
            return np.random.randint(self._n_actions)

        q_val = self._predict_q()
        return np.argmax(q_val)

    def _predict_q(self):
        # _LG.debug('Predicting Q value from NN')
        state = self._recorder.get_last_record()['state1'][None, ...]
        if luchador.get_nn_conv_format() == 'NHWC':
            state = _transpose(state)
        return self._ql.predict_action_value(state)[0]

    ###########################################################################
    # Methods for `learn`
    def learn(self, state0, action, reward, state1, terminal, info=None):
        self._n_obs += 1
        self._record(state0, action, reward, state1, terminal)
        self._train()

    def _record(self, state0, action, reward, state1, terminal):
        """Stack states and push them to recorder, then sort memory"""
        self._recorder.push(1, {
            'state0': state0, 'action': action, 'reward': reward,
            'state1': state1, 'terminal': terminal})
        self._ready = True

        cfg = self.args['record_config']
        sort_freq = cfg['sort_frequency']
        if sort_freq > 0 and self._n_obs % sort_freq == 0:
            _LG.info('Sorting Memory')
            self._recorder.sort()
            _LG.debug('Sorting Complete')

        save_freq = cfg.get('save_frequency', 0)
        if save_freq > 0 and self._n_obs % save_freq == 0:
            _LG.info('Saiving Replay Memory')
            self._save_record()

    def _save_record(self):
        save_dir = self.args['record_config']['save_dir']
        for id_, record in self._recorder.id2record.items():
            path = os.path.join(save_dir, str(id_))
            np.savez_compressed(path, **record)

    # -------------------------------------------------------------------------
    # Training
    def _train(self):
        """Schedule training"""
        cfg = self.args['training_config']
        if cfg['train_start'] < 0 or self._n_obs < cfg['train_start']:
            return

        if self._n_obs == cfg['train_start']:
            _LG.info('Starting DQN training')

        if self._n_obs % cfg['sync_frequency'] == 0:
            _LG.debug('Syncing networks')
            self._ql.sync_network()

        if self._n_obs % cfg['train_frequency'] == 0:
            error = self._train_network()
            self._n_train += 1
            self._summary_values['errors'].append(error)
            self._save_and_summarize()

    def _sample(self):
        """Sample transition from recorder and build training batch"""
        data = self._recorder.sample()
        records = data['records']
        state0 = np.asarray([r['state0'] for r in records])
        state1 = np.asarray([r['state1'] for r in records])
        reward = [r['reward'] for r in records]
        action = [r['action'] for r in records]
        terminal = [r['terminal'] for r in records]
        weights, indices = data['weights'], data['indices']
        samples = {
            'state0': state0, 'state1': state1, 'reward': reward,
            'action': action, 'terminal': terminal, 'weight': weights,
        }
        return samples, indices

    def _train_network(self):
        """Train network"""
        samples, indices = self._sample()
        if luchador.get_nn_conv_format() == 'NHWC':
            samples['state0'] = _transpose(samples['state0'])
            samples['state1'] = _transpose(samples['state1'])
        errors = self._ql.train(**samples)
        self._recorder.update(indices, np.abs(errors))
        return errors

    # -------------------------------------------------------------------------
    # Save and summarize
    def _save_and_summarize(self):
        """Save model parameter and summarize occasionally"""
        interval = self.args['save_config']['interval']
        if interval > 0 and self._n_train % interval == 0:
            _LG.info('Saving parameters')
            self._save_parameters()

        interval = self.args['summary_config']['interval']
        if interval > 0 and self._n_train % interval == 0:
            _LG.info('Summarizing Network')
            self._summarize_layer_params()
            self._summarize_layer_outputs()
            self._summarize_history()

    def _save_parameters(self):
        """Save trained parameters to file"""
        data = self._ql.get_parameters_to_serialize()
        self._saver.save(data, global_step=self._n_train)

    def _summarize_layer_params(self):
        """Summarize layer parameter statistic"""
        dataset = self._ql.get_parameters_to_summarize()
        self._summary_writer.summarize(
            summary_type='histogram',
            global_step=self._n_train, dataset=dataset)

    def _summarize_layer_outputs(self):
        """Summarize layer output"""
        samples, _ = self._sample()
        if luchador.get_nn_conv_format() == 'NHWC':
            samples['state0'] = _transpose(samples['state0'])
        dataset = self._ql.get_layer_outputs(samples['state0'])
        self._summary_writer.summarize(
            summary_type='histogram',
            global_step=self._n_train, dataset=dataset)

    def _summarize_history(self):
        """Summarize training history"""
        steps = self._summary_values['steps']
        errors = self._summary_values['errors']
        rewards = self._summary_values['rewards']
        episode = self._summary_values['episode']
        self._summary_writer.summarize(
            summary_type='histogram',
            global_step=self._n_train,
            dataset={
                'Training/Error': errors,
                'Training/Reward': rewards,
                'Training/Steps': steps,
            }
        )
        self._summary_writer.summarize(
            summary_type='scalar',
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
        self._summary_values['rewards'].append(stats['rewards'])
        self._summary_values['steps'].append(stats['steps'])
        self._summary_values['episode'] = stats['episode']
