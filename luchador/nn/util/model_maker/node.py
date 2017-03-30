"""Define Node builder"""
from __future__ import absolute_import

import logging

from ... import core
from .io import make_io_node

__all__ = ['make_node']
_LG = logging.getLogger(__name__)


def make_node(config):
    """Make Node from configuration

    Parameters
    ----------
    config : dict
        Node configuration.

        typename : str
            Name of Node class to build
        args : dict
            Constructor argument of the class
        parameters : dict
            [Optional] Configurations fof parameters to re-use in the
            resulting Node instance
        input : list or dict
            The configuration of input to the Node

    Returns
    -------
    Node
        The resulting Node isntance

    Examples
    --------
    The following configuration will create :any:``SSE`` (sum-squared error)
    between already-defined Input instance and Tensor.

    .. code-block:: yaml

        typename: SSE
        args:
          name: reconstruction_error
        input:
          target:
            typename: Input
            reuse: True
            name: input_image
          prediction:
            typename: Tensor
            name: layer8/ReLU/output
    """
    _LG.info('  Constructing: %s', config)
    if 'typename' not in config:
        raise RuntimeError('Node `typename` is not given')

    node = core.fetch_node(config['typename'])(**config.get('args', {}))

    if 'parameters' in config:
        parameters = make_io_node(config['parameters'])
        node.set_parameter_variables(**parameters)

    if 'input_config' in config:
        input_ = make_io_node(config['input_config'])
        if isinstance(input_, list):
            node(*input_)
        elif isinstance(input_, dict):
            node(**input_)
        else:
            node(input_)
    return node
