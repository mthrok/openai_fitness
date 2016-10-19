from __future__ import absolute_import

import logging
import warnings
from collections import OrderedDict

import theano
import numpy as np

from luchador.common import is_iteratable
from ..base import Session as BaseSession
from . import scope
from .wrapper import (
    Tensor,
    Operation,
)

_LG = logging.getLogger(__name__)

__all__ = ['Session']


def _get_full_class(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)

_TENSOR_CLASS_STR = _get_full_class(Tensor)
_OP_CLASS_STR = _get_full_class(Operation)


def _parse_inputs(inputs):
    inputs_ = []
    if inputs is None:
        return inputs_

    if not is_iteratable(inputs):
        inputs = [inputs]

    if isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs_.append(key.unwrap())
    elif isinstance(inputs, list):
        for key, value in inputs:
            inputs_.append(key.unwrap())
    else:
        raise ValueError(
            '`inputs` must be either dict or list of Tensor-value pair. '
            'Given: {}'.format(type(inputs)))
    return inputs_


def _parse_outputs(outputs):
    if outputs is None:
        return []
    if not is_iteratable(outputs):
        outputs = [outputs]
    return [o.unwrap() for o in outputs]


def _parse_updates(updates):
    ret = OrderedDict()
    if updates is None:
        return ret

    if not is_iteratable(updates):
        updates = [updates]

    for update in updates:
        if not isinstance(update, Operation):
            raise ValueError(
                '`updates` must be [list of] {}. Given: {}'
                .format(_OP_CLASS_STR, _get_full_class(type(update))))
        for shared_variable, new_expression in update.unwrap().items():
            ret[shared_variable] = new_expression
    return ret


def _construct_function(inputs, outputs, updates, givens):
    inputs_ = _parse_inputs(inputs)
    outputs_ = _parse_outputs(outputs)
    updates_ = _parse_updates(updates)
    return theano.function(inputs_, outputs_, updates=updates_, givens=givens)


class Session(BaseSession):
    def __init__(self, **kwargs):
        self.functions = {}

    @property
    def graph(self):
        return None

    def run(self, outputs=[], inputs={}, updates=None, givens=None, name=None):
        if name in self.functions:
            function = self.functions[name]
        else:
            function = _construct_function(inputs, outputs, updates, givens)

        if name and name not in self.functions:
            self.functions[name] = function

        values = function(*inputs.values())
        if is_iteratable(outputs):
            return values
        return values[0]

    def initialize(self):
        pass

    def close(self):
        warnings.warn('`close` does nothing in Theano backend.')
        pass

    ###########################################################################
    def load_dataset(self, dataset, cast=True):
        """Set the value of Variables with the given values

        Args:
          dataset(Dict): The keys are the names of Variables to be set, values
            are the NumPy arrays with which value are used.

          cast (Bool): If True, values are casted to the dtype of Variables.
            When False and if dtypes of Variables and dataset do not match,
            It raise TypeError.
        """
        op = OrderedDict()
        with scope.variable_scope(scope.VariableScope(reuse=True, name='')):
            for name, value in dataset.items():
                _LG.info('  Loading: {:10} {:24} {}'
                         .format(value.dtype, value.shape, name))

                variable = scope.get_variable(name=name)
                if cast:
                    value = np.array(value, dtype=variable.dtype)

                src_shape, tgt_shape = value.shape, variable.shape
                if not tgt_shape == src_shape:
                    # Theano's convolution filter shape is
                    #  [#out-channel, #in-channel, height, width]
                    # while, that of Tensorflow is
                    #  [height, width, #in-channel, #out-channel]
                    # we reshape the variable only when this condition is met
                    if (
                            len(tgt_shape) == len(src_shape) == 4 and
                            src_shape[:2] == tgt_shape[2:4] and  # h, w
                            src_shape[2:4] == tgt_shape[-3::-1]  # channels
                    ):
                        _LG.info('    Reshaping variable: {} -> {}'
                                 .format(src_shape, tgt_shape))
                        value = value.transpose((3, 2, 0, 1))
                        value = value[:, :, ::-1, ::-1]
                    else:
                        raise ValueError(
                            'Shapes are not compatible. '
                            'Model shape: {}, Value shape: {}'
                            .format(src_shape, tgt_shape)
                        )
                op[variable.unwrap()] = value
        self.run(name=None, updates=Operation(op=op))
