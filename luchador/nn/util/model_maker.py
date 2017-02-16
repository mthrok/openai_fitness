"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging

import luchador.nn
from ..model import Sequential
from .getter import get_input, get_layer, get_tensor

_LG = logging.getLogger(__name__)


def make_input(config):
    """Make input instance from configuration.

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
    >>> input1 = make_input(input_config1)

    then, to reuse the input,

    >>> input_config2 = {
    >>>     'typename': 'Input',
    >>>     'reuse': True,
    >>>     'name': 'input_state',
    >>> }
    >>> input2 = make_input(input_config2)
    >>> assert input1 is inpput2

    To reuse a Tensor from some output

    >>> input_config3 = {
    >>>     'typename': 'Tensor',
    >>>     'name': 'output_state',
    >>> }
    >>> input3 = make_input(input_config3)

    You can also create multiple Input instances

    >>> inputs = make_input([input_config2, input_config3])
    >>> assert input2 is inputs[0]
    >>> assert input3 is inputs[1]

    Returns
    -------
    [list of] ``Tensor`` or ``Input``
    """
    if isinstance(config, list):
        return [make_input(cfg) for cfg in config]

    type_ = config.get('typename', 'No `typename` is found.')
    if type_ not in ['Input', 'Tensor']:
        raise ValueError('Unexpected Input type: {}'.format(type_))

    if type_ == 'Tensor':
        return get_tensor(name=config['name'])
    # `Input` class
    if config.get('reuse'):
        return get_input(config['name'])
    return luchador.nn.backend.Input(**config['args'])


def make_sequential_model(layer_configs, input_config=None):
    """Make model instance from model configuration

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
    for config in layer_configs:
        if 'typename' not in config:
            raise RuntimeError(
                'Layer `typename` is not given in {}'.format(config))

        _LG.info('  Constructing Layer: %s', config)
        layer = get_layer(config['typename'])(**config.get('args', {}))
        model.add_layer(layer=layer, scope=config.get('scope', ''))

    if input_config:
        model(make_input(input_config))
    return model


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
    if isinstance(model_config, list):
        return [make_model(cfg) for cfg in model_config]

    _type = model_config.get('typename', 'No model type found')
    if _type == 'Sequential':
        return make_sequential_model(**model_config.get('args', {}))

    raise ValueError('Unexpected model type: {}'.format(_type))
