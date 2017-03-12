"""Utility functions for facilitating model construction"""
from __future__ import absolute_import

import logging
from collections import OrderedDict

from ... import core
from ...model import get_model
from .common import ConfigDict, parse_config

_LG = logging.getLogger(__name__)


def _make_io_node(config):
    type_ = config['typename']
    if type_ == 'Tensor':
        ret = core.get_tensor(name=config['name'])
    elif type_ == 'Variable':
        ret = core.get_variable(name=config['name'])
    elif type_ == 'Input':
        if config.get('reuse'):
            ret = core.get_input(config['name'])
        else:
            ret = core.Input(**config['args'])
    elif type_ == 'Model':
        model = get_model(config['name'])
        fetch = config['fetch']
        if fetch == 'output':
            ret = model.output
        elif fetch == 'input':
            ret = model.input
        elif fetch == 'parameter':
            ret = model.get_parameters_to_train()
    else:
        raise ValueError('Unexpected IO type: {}'.format(type_))
    return ret


def _make_io_node_recursively(config):
    if isinstance(config, ConfigDict):
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
    return _make_io_node_recursively(parse_config(config))
