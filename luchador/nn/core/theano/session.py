from __future__ import absolute_import

import logging
from collections import OrderedDict

import theano
import numpy as np

from luchador.common import is_iteratable
from ..base import session
from . import scope, wrapper


_LG = logging.getLogger(__name__)

__all__ = ['Session']


def _get_full_class(cls):
    return '{}.{}'.format(cls.__module__, cls.__name__)

_TENSOR_CLASS_STR = _get_full_class(wrapper.Tensor)
_OP_CLASS_STR = _get_full_class(wrapper.Operation)


def _parse_inputs(inputs):
    inputs_ = []
    if inputs is None:
        return inputs_

    if not is_iteratable(inputs):
        inputs = [inputs]

    try:
        for key in inputs:
            inputs_.append(key.unwrap())
    except Exception:
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
        if not isinstance(update, wrapper.Operation):
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


class Session(session.BaseSession):
    """Handles operations and computations in similar way as Tensorflow session
    """
    def __init__(self, **kwargs):
        super(Session, self).__init__()
        self.functions = {}

    def run(self, outputs=None, inputs=None,
            updates=None, givens=None, name=None):
        outputs = outputs if outputs else []
        inputs = inputs if inputs else {}
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
        """Compatibility for TF backend. Does nothing in Theano backend"""
        pass

    def close(self):
        """Compatibility for TF backend. Does nothing in Theano backend"""
        pass

    ###########################################################################
    def load_dataset(self, dataset, cast=True, strict=True):
        """Set the value of Variables with the given values

        Args:
          dataset(Dict): The keys are the names of Variables to be set, values
            are the NumPy arrays with which value are used.

          cast (Bool): If True, values are casted to the dtype of Variables.
            When False and if dtypes of Variables and dataset do not match,
            It raise TypeError.

          strict (Bool): When True, if dataset contains a value for Variable
            which is not defined, then ValueError exception is raised.
            Otherwise it will be skipped.
        """
        ops = OrderedDict()
        with scope.variable_scope(scope.VariableScope(reuse=True, name='')):
            for name, value in dataset.items():
                try:
                    variable = scope.get_variable(name=name)
                    _LG.info('  Loading: %10s %-24s %s',
                             value.dtype, value.shape, name)
                except ValueError:
                    if strict:
                        raise
                    _LG.info('  Variable `%s` does not exist.', name)
                    continue

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
                        _LG.info('    Reshaping variable: %s -> %s',
                                 src_shape, tgt_shape)
                        value = value.transpose((3, 2, 0, 1))
                        value = value[:, :, ::-1, ::-1]
                    else:
                        raise ValueError(
                            'Shapes are not compatible. '
                            'Model shape: {}, Value shape: {}'
                            .format(src_shape, tgt_shape)
                        )
                ops[variable.unwrap()] = value
        self.run(name=None, updates=wrapper.Operation(op=ops))
