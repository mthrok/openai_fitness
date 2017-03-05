"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

from .. import core
from ..model import Sequential, Container

_LG = logging.getLogger(__name__)


###############################################################################
class _ConfigDict(OrderedDict):
    pass


def _parse_config(config):
    """Mark valid configurations as ConfigDict class

    so as to differentiate them from ordinal dict"""
    if isinstance(config, list):
        return [_parse_config(cfg) for cfg in config]
    if isinstance(config, dict):
        _config = _ConfigDict() if 'typename' in config else OrderedDict()
        for key, value in config.items():
            _config[key] = _parse_config(value)
        return _config
    return config


###############################################################################
def _make_io_node(config):
    type_ = config['typename']
    if type_ == 'Tensor':
        ret = core.get_tensor(name=config['name'])
    elif type_ == 'Variable':
        # TODO: Add make_variable here?
        ret = core.get_variable(name=config['name'])
    elif type_ == 'Input':
        if config.get('reuse'):
            ret = core.get_input(config['name'])
        else:
            ret = core.Input(**config['args'])
    else:
        raise ValueError('Unexpected IO type: {}'.format(type_))
    return ret


def _make_io_node_recursively(config):
    if isinstance(config, _ConfigDict):
        return _make_io_node(config)
    if isinstance(config, list):
        return [make_io_node(cfg) for cfg in config]
    if isinstance(config, dict):
        ret = OrderedDict()
        for key, value in config.items():
            ret[key] = make_io_node(value)
        return ret

    raise ValueError('Invalid IO config: {}'.format(config))


def make_io_node(config):
    """Make/fetch ``Input``/``Tensor`` instances from configuration.

    This function was itroduced to facilitate model construction from YAML.

    Parameters
    ----------
    config : [list of] dict
        typename : str
            Either ``Input`` or ``Tensor``.
        reuse : bool
            When typename is ``Input``, this value determines either the to
            reuse an existing instance of Input class or, to create new
            instance.
        args : dict
            Constructor agruments for creating new Input instance.
            See :any:`BaseInput` for detail.
        name : str
            When reusing ``Input`` or ``Tensor``, existing instance is
            retrieved using this name.

    Notes
    -----
    When fetching ``Tensor`` or an existing ``Input`` isntance with reuse, they
    must be instantiated before this function is called.

    Examples
    --------
    To create new Input

    >>> input_config1 = {
    >>>     'typename': 'Input',
    >>>     'args': {
    >>>         'dtype': 'uint8',
    >>>         'shape': [32, 5],
    >>>         'name': 'input_state',
    >>>     },
    >>> }
    >>> input1 = make_io_node(input_config1)

    then, to reuse the input,

    >>> input_config2 = {
    >>>     'typename': 'Input',
    >>>     'reuse': True,
    >>>     'name': 'input_state',
    >>> }
    >>> input2 = make_io_node(input_config2)
    >>> assert input1 is inpput2

    To reuse a Tensor from some output

    >>> input_config3 = {
    >>>     'typename': 'Tensor',
    >>>     'name': 'output_state',
    >>> }
    >>> input3 = make_io_node(input_config3)

    You can also create multiple Input instances

    >>> inputs = make_io_node([input_config2, input_config3])
    >>> assert input2 is inputs[0]
    >>> assert input3 is inputs[1]

    >>> inputs = make_io_node({
    >>>     'input2': input_config2, 'input3': input_config3})
    >>> assert input2 is inputs['input2']
    >>> assert input3 is inputs['input3']

    Returns
    -------
    [list of] ``Tensor`` or ``Input``
    """
    return _make_io_node_recursively(_parse_config(config))


###############################################################################
def make_layer(layer_config):
    """Make Layer instance

    Parameters
    ----------
    layer_config : dict
        typename : str
            Name of Layer class to instanciate
        args : dict
            Constructor arguments for the Layer class
        parameters: dict
            Parameter Variables to resuse. Keys are parameter keys
            and values are configuration acceptable by :func:`make_io_node`.

    Returns
    -------
    layer
        Layer object
    """
    _LG.info('  Constructing Layer: %s', layer_config)
    if 'typename' not in layer_config:
        raise RuntimeError('Layer `typename` is not given')

    layer = core.get_layer(
        layer_config['typename'])(**layer_config.get('args', {}))

    if 'parameters' in layer_config:
        parameters = _make_io_node_recursively(layer_config['parameters'])
        layer.set_parameter_variables(**parameters)
    return layer


###############################################################################
def _make_sequential_model(layer_configs, input_config=None):
    """Make Sequential model instance from model configuration

    Parameters
    ----------
    layer_configs : list
        ``Layer`` configuration.

    input_config : dict or None
        ``Input`` configuraiton to the model. If give, layers are built on the
        input spcified by this configuration, otherwise, model is returned
        unbuilt.

    Returns
    -------
    Model
        Resulting model
    """
    model = Sequential()
    if input_config:
        tensor = _make_io_node_recursively(input_config)
    for config in layer_configs:
        layer = make_layer(config)
        if input_config:
            tensor = layer(tensor)
        model.add_layer(layer)
    return model


def _make_container_model(input_config, model_configs, output_config=None):
    """Make ``Container`` model from model configuration

    Parameters
    ----------
    input_config : [list of] dict
        See :any::make_io_node

    model_config : list
        Model configuration.

    output_config : [list of] dict
        See :any::make_io_node

    Returns
    -------
    Model
        Resulting model
    """
    model = Container()
    model.input = _make_io_node_recursively(input_config)
    for conf in model_configs:
        _LG.info('Building Model: %s', conf.get('name', 'No name defined'))
        model.add_model(conf['name'], make_model(conf))

    if output_config:
        model.output = _make_io_node_recursively(output_config)
    return model


def _make_model(model_config):
    _type = model_config.get('typename', 'No model type found')
    if _type == 'Sequential':
        return _make_sequential_model(**model_config.get('args', {}))
    if _type == 'Container':
        return _make_container_model(**model_config.get('args', {}))
    raise ValueError('Unexpected model type: {}'.format(_type))


def _make_model_recursively(model_config):
    if isinstance(model_config, _ConfigDict):
        return _make_model(model_config)
    if isinstance(model_config, list):
        return [_make_model_recursively(cfg) for cfg in model_config]
    if isinstance(model_config, dict):
        ret = OrderedDict()
        for key, value in model_config.items():
            ret[key] = _make_model_recursively(value)
        return ret

    raise ValueError('Invalid model config: {}'.format(model_config))


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_config : [list of] dict
        Model configuration in dict or list of configurations.

    Returns
    -------
    [list of] Model
        Resulting model[s]
    """
    model_config = _parse_config(model_config)
    return _make_model_recursively(model_config)
