"""Module to define utility functions used in luchador.nn module

This module is expected to be loaded before backend is loaded,
thus should not cause cyclic import.
"""
from __future__ import absolute_import

import os
import logging
import StringIO

import ruamel.yaml as yaml

from luchador.util import get_subclasses
import luchador.nn

from .model.sequential import make_sequential_model

__all__ = ['make_model', 'get_model_config', 'get_grad']


###############################################################################
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')
_LG = logging.getLogger(__name__)


def _get_input():
    for class_ in get_subclasses(luchador.nn.base.wrapper.BaseWrapper):
        if class_.__name__ == 'Input':
            return class_
    raise ValueError('`Input` class is not defined in current backend.')


def _make_input(config):
    if isinstance(config, list):
        return [_make_input(cfg) for cfg in config]

    type_ = config['typename']
    if type_ == 'Input':
        input_ = _get_input()(**config['args'])
    elif type_ == 'Tensor':
        input_ = luchador.nn.get_tensor(name=config['name'])
    else:
        raise ValueError('Unexpected Input type: {}'.format(type_))
    return input_


def make_model(model_config):
    """Make model from model configuration

    Parameters
    ----------
    model_config : dict or list
        model configuration in dict or list of configurations.

    Returns
    -------
    Model or list of Model
        Resulting models
    """
    if isinstance(model_config, list):
        return [make_model(cfg) for cfg in model_config]

    _type = model_config['model_type']
    if _type == 'Sequential':
        model = make_sequential_model(model_config['layer_configs'])
    else:
        raise ValueError('Unexpected model type: {}'.format(_type))

    if model_config.get('input'):
        input_ = _make_input(model_config['input'])
        model(input_)
    return model


def get_model_config(model_name, **parameters):
    """Load pre-defined model configurations

    Parameters
    ----------
    model_name : str
        Model name or path to YAML file

    parameters
        Parameter for model config

    Returns
    -------
    JSON-compatible object
        Model configuration.
    """
    file_name = model_name
    if not file_name.endswith('.yml'):
        file_name = '{}.yml'.format(file_name)

    if os.path.isfile(file_name):
        file_path = file_name
    elif os.path.isfile(os.path.join(_DATA_DIR, file_name)):
        file_path = os.path.join(_DATA_DIR, file_name)
    else:
        raise ValueError(
            'No model definition file ({}) found.'.format(file_name))

    with open(file_path, 'r') as file_:
        model_text = file_.read()

    if parameters:
        model_text = model_text.format(**parameters)

    model_text = StringIO.StringIO(model_text)
    return yaml.safe_load(model_text)


###############################################################################
def get_grad(var):
    """Fetch gradient tensor corresponding to the given Variable

    In optimizers, gradient tensors are registered in global list of tensors,
    following the naming pattern `<scope>/<variable_name>_grad`.

    This function automatically build such name from the given Variable and
    the current scope.

    To properly fetch the corresponding gradient Tensor, this function
    must be called in the scope where gradient Tensor was defined.

    Examples
    --------
    >>> from luchador import nn
    >>> x = nn.get_variable(shape=(), name='x')
    >>> # Variable x is registered with name 'x'
    >>> y = x * x
    >>> sgd = nn.optimizer.SGD(learning_rate=0.1)
    >>> with nn.variable_scope('optimization'):
    >>>    sgd.minimize(loss=y, wrt=x)
    >>>    # dydx is registered with name '/optimization/x_grad'
    >>>    dydx2 = nn.get_grad_tensor(x)
    >>>    assert dydx1 is dydx2

    Parameters
    ----------
    var : Variable
        Variable object of which grad is retrieved.

    Returns
    -------
    Tensor
        Tensor object which is a gradient of given Variable
    """
    return luchador.nn.get_tensor('{}_grad'.format(var.name))
