"""Module for building neural Q learning network"""
from __future__ import division
from __future__ import absolute_import

import logging
from collections import OrderedDict

import luchador.util
from luchador import nn

_LG = logging.getLogger(__name__)

__all__ = ['DeepQLearning']


def _validate_q_learning_config(
        min_reward=None, max_reward=None,
        min_delta=None, max_delta=None, **_):
    if (min_reward and not max_reward) or (max_reward and not min_reward):
        raise ValueError(
            'When clipping reward, both `min_reward` '
            'and `max_reward` must be provided.')
    if (min_delta and not max_delta) or (max_delta and not min_delta):
        raise ValueError(
            'When clipping reward, both `min_delta` '
            'and `max_delta` must be provided.')


class DeepQLearning(luchador.util.StoreMixin, object):
    """Build Q-learning network and optimization operations

    Parameters
    ----------
    discout_rate : float
        Discount rate for computing future reward. Valid value range is
        (0.0, 1.0)

    scale_reward : number or None
        When given, reward is divided by this number before applying min/max
        threashold

    min_reward : number or None
        When given, clip reward after scaling.

    max_reward : number or None
        See `min_reward`.

    min_delta : number or None
        When given, error between predicted Q and target Q is clipped with
        this value.

    max_delta : number or None
        See `max_reward`
    """
    def __init__(
            self, q_learning_config, optimizer_config,
            saver_config, summary_writer_config, session_config):
        self._store_args(
            q_learning_config=q_learning_config,
            optimizer_config=optimizer_config,
            summary_writer_config=summary_writer_config,
            saver_config=saver_config,
            session_config=session_config
        )
        self.vars = {
            'state0': None,
            'action': None,
            'reward': None,
            'state1': None,
            'terminal': None,
            'action_value_0': None,
            'target_q': None,
            'error': None,
        }
        self.models = {
            'pre_trans': None,
            'post_trans': None,
        }
        self.ops = {
            'sync': None,
        }
        self.optimizer = None
        self.session = None
        self.saver = nn.Saver(**saver_config)
        self.summary_writer = nn.SummaryWriter(**summary_writer_config)

        self.n_trainings = 0

    def _validate_args(self, q_learning_config=None, **_):
        if q_learning_config is not None:
            _validate_q_learning_config(**q_learning_config)

    def __call__(self, q_network_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network_maker(function): Model factory function which are called
            without any arguments and return Model object
        """
        self.build(q_network_maker)

    def build(self, model_maker):
        """Build computation graph (error and sync ops) for Q learning

        Args:
          q_network(TFModel): TFModel which represetns Q network.
            Model must be pre-built since this function needs shape
            information.
        """
        with nn.variable_scope('pre_trans'):
            self.models['pre_trans'] = model_maker()
            self.vars['state0'] = self.models['pre_trans'].input
        with nn.variable_scope('post_trans'):
            self.models['post_trans'] = model_maker()
            self.vars['state1'] = self.models['post_trans'].input
        with nn.variable_scope('target_q_value'):
            self._build_target_q_value()
        with nn.variable_scope('sync'):
            self._build_sync_op()
        with nn.variable_scope('error'):
            self._build_error()

        self._build_optimization_op()
        self._init_session()
        self._init_summary_writer()
        return self

    ###########################################################################
    def _build_target_q_value(self):
        self.vars['terminal'] = terminal = nn.Input(
            shape=(None,), name='terminal')
        self.vars['reward'] = reward = nn.Input(
            shape=(None,), name='rewards')
        self.vars['action_value_0'] = self.models['pre_trans'].output

        cfg = self.args['q_learning_config']

        if 'scale_reward' in cfg:
            reward = reward / cfg['scale_reward']
        if 'min_reward' in cfg and 'max_reward' in cfg:
            min_val, max_val = cfg['min_reward'], cfg['max_reward']
            reward = reward.clip(min_value=min_val, max_value=max_val)

        max_q = self.models['post_trans'].output.max(axis=1)
        discounted = max_q * cfg['discount_rate']
        target_q = reward + (1.0 - terminal) * discounted

        n_actions = self.models['pre_trans'].output.shape[1]
        target_q = target_q.reshape([-1, 1]).tile([1, n_actions])
        self.vars['target_q'] = target_q

    def _build_error(self):
        self.vars['action'] = action = nn.Input(
            shape=(None,), dtype='uint8', name='action')

        cfg = self.args['q_learning_config']
        min_, max_ = cfg.get('min_delta', None), cfg.get('max_delta', None)
        sse2 = nn.cost.SSE2(min_delta=min_, max_delta=max_, elementwise=True)
        error = sse2(self.vars['target_q'], self.vars['action_value_0'])

        n_actions = self.models['pre_trans'].output.shape[1]
        mask = action.one_hot(n_classes=n_actions)

        self.vars['error'] = (mask * error).mean()

    def _build_optimization_op(self):
        cfg = self.args['optimizer_config']
        self.optimizer = nn.get_optimizer(cfg['name'])(**cfg['args'])
        wrt = self.models['pre_trans'].get_parameter_variables()
        self.ops['optimize'] = self.optimizer.minimize(
            self.vars['error'], wrt=wrt)

    def _build_sync_op(self):
        src_vars = self.models['pre_trans'].get_parameter_variables()
        tgt_vars = self.models['post_trans'].get_parameter_variables()
        self.ops['sync'] = nn.build_sync_op(src_vars, tgt_vars, name='sync')

    ###########################################################################
    def _init_session(self):
        cfg = self.args['session_config']
        self.session = nn.Session()
        if cfg.get('parameter_file'):
            _LG.info('Loading paramter from %s', cfg['parameter_file'])
            self.session.load_from_file(cfg['parameter_file'])
        else:
            self.session.initialize()

    ###########################################################################
    def predict_action_value(self, state0):
        return self.session.run(
            outputs=self.vars['action_value_0'],
            inputs={self.vars['state0']: state0},
            name='action_value0',
        )

    def sync_network(self):
        self.session.run(updates=self.ops['sync'], name='sync')

    def train(self, state0, action, reward, state1, terminal):
        updates = self.models['pre_trans'].get_update_operations()
        updates += [self.ops['optimize']]
        self.n_trainings += 1
        return self.session.run(
            outputs=self.vars['error'],
            inputs={
                self.vars['state0']: state0,
                self.vars['action']: action,
                self.vars['reward']: reward,
                self.vars['state1']: state1,
                self.vars['terminal']: terminal,
            },
            updates=updates,
            name='minibatch_training',
        )

    ###########################################################################
    def save(self):
        """Save network parameter to file"""
        params = (
            self.models['pre_trans'].get_parameter_variables() +
            self.optimizer.get_parameter_variables())
        params_val = self.session.run(outputs=params, name='pre_trans_params')
        self.saver.save(OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ]), global_step=self.n_trainings)

    ###########################################################################
    def _init_summary_writer(self):
        """Initialize SummaryWriter and create set of summary operations"""
        if self.session.graph:
            self.summary_writer.add_graph(self.session.graph)

        params = self.models['pre_trans'].get_parameter_variables()
        outputs = self.models['pre_trans'].get_output_tensors()
        self.summary_writer.register(
            'histogram', tag='params',
            names=['/'.join(v.name.split('/')[1:]) for v in params])
        self.summary_writer.register(
            'histogram', tag='outputs',
            names=['/'.join(v.name.split('/')[1:]) for v in outputs])
        self.summary_writer.register(
            'histogram', tag='training',
            names=['Training/Error', 'Training/Reward', 'Training/Steps']
        )
        self.summary_writer.register_stats(['Error', 'Reward', 'Steps'])
        self.summary_writer.register('scalar', ['Episode'])

    def summarize_layer_params(self):
        """Summarize paramters of each layer"""
        params = self.models['pre_trans'].get_parameter_variables()
        params_vals = self.session.run(outputs=params, name='pre_trans_params')
        params_data = {
            '/'.join(v.name.split('/')[1:]): val
            for v, val in zip(params, params_vals)
        }
        self.summary_writer.summarize(self.n_trainings, params_data)

    def summarize_layer_outputs(self, state):
        """Summarize outputs from each layer

        Parameters
        ----------
        state : NumPy ND Array
            Input to model0 (pre-transition model)
        """
        outputs = self.models['pre_trans'].get_output_tensors()
        output_vals = self.session.run(
            outputs=outputs,
            inputs={self.vars['state0']: state},
            name='pre_trans_outputs'
        )
        output_data = {
            '/'.join(v.name.split('/')[1:]): val
            for v, val in zip(outputs, output_vals)
        }
        self.summary_writer.summarize(self.n_trainings, output_data)

    def summarize_stats(self, episode, errors, rewards, steps):
        """Summarize training history"""
        self.summary_writer.summarize(
            global_step=self.n_trainings, tag='training',
            dataset=[errors, rewards, steps]
        )
        self.summary_writer.summarize(
            global_step=self.n_trainings, dataset={'Episode': episode}
        )
        if rewards:
            self.summary_writer.summarize_stats(
                global_step=self.n_trainings, dataset={'Reward': rewards}
            )
        if errors:
            self.summary_writer.summarize_stats(
                global_step=self.n_trainings, dataset={'Error': errors}
            )
        if steps:
            self.summary_writer.summarize_stats(
                global_step=self.n_trainings, dataset={'Steps': steps}
            )
