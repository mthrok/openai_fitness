from __future__ import absolute_import

import logging

import tensorflow as tf
from tensorflow import (
    Session as TFSession,
)

from luchador.common import is_iteratable
from ..base import Session as BaseSession
from . import scope
from .wrapper import (
    Tensor,
    Variable,
    Operation,
)

_LG = logging.getLogger(__name__)

__all__ = ['Session']


def _get_full_class(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)

_TENSOR_CLASS_STR = _get_full_class(Tensor)
_OP_CLASS_STR = _get_full_class(Operation)


def _parse_outputs(outputs):
    outputs_ = []
    if outputs is None:
        return outputs_

    if not is_iteratable(outputs):
        outputs = [outputs]

    for output in outputs:
        if not (isinstance(output, Tensor) or isinstance(output, Variable)):
            raise ValueError(
                '`outputs` must be [list of] {}. Given: {}'
                .format(_TENSOR_CLASS_STR, _get_full_class(type(output))))
        outputs_.append(output.get())
    return outputs_


def _parse_updates(updates):
    ret = []
    if not updates:
        return ret

    if not is_iteratable(updates):
        updates = [updates]

    for update in updates:
        if not isinstance(update, Operation):
            raise ValueError(
                '`updates` must be [list of] {}. Given: {}'
                .format(_OP_CLASS_STR, _get_full_class(type(update))))

        ret.append(update.get())
    return ret


def _construct_fetches(outputs, updates):
    return _parse_outputs(outputs) + _parse_updates(updates)


def _construct_feed_dict(inputs, givens):
    feed_dict = {}
    if not inputs:
        pass
    elif isinstance(inputs, dict):
        for key, value in inputs.items():
            feed_dict[key.get()] = value
    elif isinstance(inputs, list):
        for key, value in inputs:
            feed_dict[key.get()] = value
    else:
        raise ValueError(
            '`inputs` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(inputs)))

    if not givens:
        pass
    elif isinstance(givens, dict):
        for key, value in givens.items():
            feed_dict[key.get()] = value
    elif isinstance(givens, list):
        for key, value in givens:
            feed_dict[key.get()] = value
    else:
        raise ValueError(
            '`givens` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(givens)))
    return feed_dict


class Session(BaseSession):
    def __init__(self, graph=None, config=None, **kwargs):
        self.session = TFSession('', graph, config)

    @property
    def graph(self):
        return self.session.graph

    def run(self, outputs=[], inputs={}, updates=None, givens=None, name=None):
        """

        Args:
          outputs (list of Tensors): Tensors of which values are fetched

          inputs (dict): Keys are the input Tensors required to compute values
                         of output Tensors. Values are actual values to feed
                         to Tensors.

          updates (Operation or list of Operations):

          givens (dict):

          name (str): Not used. Compatibility for theano backend
        """
        fetches = _construct_fetches(outputs, updates)
        feed_dict = _construct_feed_dict(inputs, givens)
        values = self.session.run(fetches, feed_dict=feed_dict)
        if is_iteratable(outputs):
            return values[:len(outputs)]
        return values[0]

    def close(self):
        return self.session.close()

    def initialize(self):
        self.session.run(tf.initialize_all_variables())

    ###########################################################################
    def load_dataset(self, dataset, cast=True):
        """Set the value of Variables with the given values

        Args:
          dataset(Dict): The keys are the names of Variables to be set, values
            are the NumPy arrays with which value are used.

          cast (Bool): Not used in Tensorflow backend as it casts dtype
            internally.
        """
        op = []
        with scope.variable_scope(scope.VariableScope(reuse=True, name='')):
            for name, value in dataset.items():
                _LG.info('  Loading: {:10} {:24} {}'
                         .format(value.dtype, value.shape, name))

                variable = scope.get_variable(name=name)
                src_shape = tuple(value.shape)
                tgt_shape = tuple(variable.shape)
                if not tgt_shape == src_shape:
                    # Tensorflow is
                    #  [height, width, #in-channel, #out-channel]
                    # while Theano's Conv2D Filter shape is
                    #  [#out-channel, #in-channel, height, width]
                    # we reshape the variable only when this condition is met
                    if (
                            len(tgt_shape) == len(src_shape) == 4 and
                            src_shape[2:4] == tgt_shape[:2] and  # h, w
                            src_shape[:2] == tgt_shape[:1:-1]  # channels
                    ):
                        value = value.transpose((2, 3, 1, 0))
                    else:
                        raise ValueError(
                            'Shapes are not incompatible. '
                            'Model shape: {}, Value shape: {}'
                            .format(src_shape, tgt_shape)
                        )
                op.append(variable.get().assign(value))
        self.run(name=None, updates=Operation(tf.group(*op)))
