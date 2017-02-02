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


def _make_model(model_def, scope):
    with nn.variable_scope(scope):
        model = nn.make_model(model_def)
        state = model.input
        action_value = model.output
    return model, state, action_value


def _build_sync_op(src_model, tgt_model, scope):
    with nn.variable_scope(scope):
        src_vars = src_model.get_parameter_variables()
        tgt_vars = tgt_model.get_parameter_variables()
        return nn.build_sync_op(src_vars, tgt_vars, name='sync')


class DeepQLearning(luchador.util.StoreMixin, object):
    """Build Q-learning network and optimization operations

    Parameters
    ----------
    model_config : dict
        Configuration for model definition.

        name : str
            The name of network model or path to model definition file.
        initial_parameter : str
            Path to HDF5 file which contain the initial parameter values.
        input_channel, input_height, input_width : int
            The shape of input to the network

    q_learning_config : dict
        Configuration for building target Q value.

        discout_rate : float
            Discount rate for computing future reward. Valid value range is
            (0.0, 1.0)
        scale_reward : number or None
            When given, reward is divided by this number before applying
            min/max threashold
        min_reward : number or None
            When given, clip reward after scaling.
        max_reward : number or None
            See `min_reward`.

    cost_config : dict
        Configuration for defining error between predicted Q and target Q

        name: str
            The name of cost class. See :py:mod:`luchador.nn.base.cost`
            for the list of available costs.
        args : dict
            Configuration for the cost class

    optimizer_config : dict
        Configuration for optimizer

        name: str
            The name of cost class. See :py:mod:`luchador.nn.base.optimizer`
            for the list of available classes.
        args : dict
            Configuration for the optimizer class

    saver_config : dict
        Constructor arguments for :class:`luchador.nn.saver.Saver`

    summary_writer_config : dict
        Constructor arguments for :class:`luchador.nn.saver.Saver`
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self, model_config, q_learning_config, cost_config,
            optimizer_config, saver_config, summary_writer_config):
        self._store_args(
            model_config=model_config,
            q_learning_config=q_learning_config,
            cost_config=cost_config,
            optimizer_config=optimizer_config,
            summary_writer_config=summary_writer_config,
            saver_config=saver_config,
        )
        self.vars = None
        self.models = None
        self.ops = None
        self.optimizer = None
        self.session = None
        self.saver = None
        self.summary_writer = None
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

    def build(self, n_actions):
        """Build computation graph (error and sync ops) for Q learning

        Parameters
        ----------
        n_actions: int
            The number of available actions in the environment.
        """
        # pylint: disable=too-many-locals
        model_def = self._gen_model_def(n_actions)
        model_0, state_0, action_value_0 = _make_model(model_def, 'pre_trans')
        model_1, state_1, action_value_1 = _make_model(model_def, 'post_trans')
        sync_op = _build_sync_op(model_0, model_1, 'sync')

        with nn.variable_scope('target_q_value'):
            reward = nn.Input(shape=(None,), name='rewards')
            terminal = nn.Input(shape=(None,), name='terminal')
            target_q = self._build_target_q_value(
                action_value_1, reward, terminal)

        with nn.variable_scope('error'):
            action_0 = nn.Input(shape=(None,), dtype='uint8', name='action_0')
            error = self._build_error(target_q, action_value_0, action_0)

        self._init_optimizer()
        optimize_op = self.optimizer.minimize(
            error, wrt=model_0.get_parameter_variables())
        self._init_session()
        self._init_summary_writer(model_0)
        self._init_saver()

        self.models = {
            'model_0': model_0,
            'model_1': model_1,
        }
        self.vars = {
            'state_0': state_0,
            'state_1': state_1,
            'action_value_0': action_value_0,
            'action_value_1': action_value_1,
            'action_0': action_0,
            'reward': reward,
            'terminal': terminal,
            'target_q': target_q,
            'error': error,
        }
        self.ops = {
            'sync': sync_op,
            'optimize': optimize_op,
        }

    def _gen_model_def(self, n_actions):
        cfg = self.args['model_config']
        fmt = luchador.get_nn_conv_format()
        w, h, c = cfg['input_width'], cfg['input_height'], cfg['input_channel']
        shape = (
            '[null, {}, {}, {}]'.format(h, w, c) if fmt == 'NHWC' else
            '[null, {}, {}, {}]'.format(c, h, w)
        )
        return nn.get_model_config(
            cfg['name'], n_actions=n_actions, input_shape=shape)

    def _build_target_q_value(self, action_value_1, reward, terminal):
        config = self.args['q_learning_config']
        # Clip rewrads
        if 'scale_reward' in config:
            reward = reward / config['scale_reward']
        if 'min_reward' in config and 'max_reward' in config:
            min_val, max_val = config['min_reward'], config['max_reward']
            reward = reward.clip(min_value=min_val, max_value=max_val)

        # Build Target Q
        max_q = action_value_1.max(axis=1)
        discounted_q = max_q * config['discount_rate']
        target_q = reward + (1.0 - terminal) * discounted_q

        n_actions = action_value_1.shape[1]
        target_q = target_q.reshape([-1, 1]).tile([1, n_actions])
        return target_q

    def _build_error(self, target_q, action_value_0, action):
        config = self.args['cost_config']
        sse2 = nn.get_cost(config['name'])(elementwise=True, **config['args'])
        error = sse2(target_q, action_value_0)
        mask = action.one_hot(n_classes=action_value_0.shape[1])
        return (mask * error).mean()

    ###########################################################################
    def _init_saver(self):
        config = self.args['saver_config']
        if config.get('output_dir'):
            self.saver = nn.Saver(**config)

    def _init_optimizer(self):
        cfg = self.args['optimizer_config']
        self.optimizer = nn.get_optimizer(cfg['name'])(**cfg['args'])

    def _init_session(self):
        cfg = self.args['model_config']
        self.session = nn.Session()
        if cfg.get('initial_parameter'):
            _LG.info('Loading parameters from %s', cfg['initial_parameter'])
            self.session.load_from_file(cfg['initial_parameter'])
        else:
            self.session.initialize()

    def _init_summary_writer(self, model_0):
        """Initialize SummaryWriter and create set of summary operations"""
        config = self.args['summary_writer_config']
        if config.get('output_dir'):
            self.summary_writer = nn.SummaryWriter(**config)
        else:
            return

        if self.session.graph:
            self.summary_writer.add_graph(self.session.graph)

        params = model_0.get_parameter_variables()
        outputs = model_0.get_output_tensors()
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

    ###########################################################################
    def predict_action_value(self, state):
        """Predict action values

        Parameters
        ----------
        state : NumPy ND Array
            Environment state

        Returns
        -------
        NumPy ND Array
            Action values
        """
        return self.session.run(
            outputs=self.vars['action_value_0'],
            inputs={self.vars['state_0']: state},
            name='action_value0',
        )

    def sync_network(self):
        """Synchronize parameters of model_1 with those of model_0"""
        self.session.run(updates=self.ops['sync'], name='sync')

    def train(self, state_0, action_0, reward, state_1, terminal):
        """Train model network

        Parameters
        ----------
        state_0 : NumPy ND Array
            Environment states before taking actions

        action_0 : NumPy ND Array
            Actions taken in state_0

        reward : NumPy ND Array
            Rewards obtained by taking the action_0.

        state_1 : NumPy ND Array
            Environment states after action_0 are taken

        terminal : NumPy ND Array
            Flags for marking corresponding states in state_1 are
            terminal states.

        Returns
        -------
        NumPy ND Array
            Mean error between Q prediction and target Q
        """
        updates = self.models['model_0'].get_update_operations()
        updates += [self.ops['optimize']]
        self.n_trainings += 1
        return self.session.run(
            outputs=self.vars['error'],
            inputs={
                self.vars['state_0']: state_0,
                self.vars['action_0']: action_0,
                self.vars['reward']: reward,
                self.vars['state_1']: state_1,
                self.vars['terminal']: terminal,
            },
            updates=updates,
            name='minibatch_training',
        )

    ###########################################################################
    def save(self):
        """Save network parameter to file"""
        params = (
            self.models['model_0'].get_parameter_variables() +
            self.optimizer.get_parameter_variables()
        )
        params_val = self.session.run(outputs=params, name='save_params')
        self.saver.save(OrderedDict([
            (var.name, val) for var, val in zip(params, params_val)
        ]), global_step=self.n_trainings)

    ###########################################################################
    def summarize_layer_params(self):
        """Summarize paramters of each layer"""
        params = self.models['model_0'].get_parameter_variables()
        params_vals = self.session.run(outputs=params, name='model_0_params')
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
            Input to model_0 (pre-transition model)
        """
        outputs = self.models['model_0'].get_output_tensors()
        output_vals = self.session.run(
            outputs=outputs,
            inputs={self.vars['state_0']: state},
            name='model_0_outputs'
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
