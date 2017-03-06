"""Define AnonymousLayer classes"""
from __future__ import division
from __future__ import absolute_import

import logging

from ...base import BaseLayer
from ...base.scope import variable_scope
from ...backend import wrapper, ops
from .. import random

__all__ = ['Anonymous']
_LG = logging.getLogger(__name__)
# pylint: disable=abstract-method


def _parse_input_tensors(input_tensor, *args, **kwargs):
    if input_tensor:
        if kwargs:
            raise ValueError(
                'Anonymouse layer accepsts either positional parameters '
                'or keyword arguments, not both.')
        if args:
            return (input_tensor,) + args
        return input_tensor
    return kwargs


def _get_safe_function(input_tensor, *args, **kwargs):
    if args and kwargs:
        raise ValueError(
            'Anonymouse layer accepsts either positional parameters '
            'or keyword arguments, not both.')
    maps = {
        'x': _parse_input_tensors(input_tensor, *args, **kwargs),
        'True': True,
        'False': False,
        'NormalRandom': random.NormalRandom,
        'UniformRandom': random.UniformRandom,
    }
    for key in ops.__all__:
        maps[key] = getattr(ops, key)
    return maps


class Anonymous(BaseLayer):
    """Run externally-provided computation on input tensor

    Parameters
    ----------
    exp : str
        String expression of computation. In this expression, ``x`` represents
        input arguments to ``build`` method. This ``exp`` is passed to ``eval``
        and the resulting value is returned.
    """
    def __init__(self, exp, name='Anonymous'):
        super(Anonymous, self).__init__(name=name, exp=exp)

    def __call__(self, input_tensor=None, *args, **kwargs):
        """Convenience method to call `build`"""
        return self.build(input_tensor, *args, **kwargs)

    def build(self, input_tensor=None, *args, **kwargs):
        """Build Anonymous layer

        For flexiblility purpose, this layer can accept multiple input, but
        there is a restriction on how the inputs are supplied. It must be
        either single positional argument, list arguments or key word
        arguments. Depending on how parameters are provided, ``x`` in
        anonymous function can be either single wrapper, list or dict. See
        example for detail.

        Parameters
        ----------
        input_tensor : Tensor
            Parameter which represents ``x`` when input is single input

        *args : Tensors
            Parameters which represents ``x`` when inputs are list

        **kwargs : Tensors
            Parameters which represents ``x`` when inputs are dict

        Returns
        -------
        Tensor
            Tensor represents the result of evaluating ``exp``

        Examples
        --------
        When input to ``build`` is single input, ``x`` in `exp`` represents
        this arguments

        >>> x = Input(shape=(3, 4), name='in0')
        >>> anon = Anonymous(exp='x', name='anon0')
        >>> # This layer just returns the input Tensor as it is
        >>> y = anon(x)
        >>> # y == x

        When input to ``build`` is list argument, ``x`` in `exp`` represents
        Tensors passed as list.

        >>> x = [
        >>>     Input(shape=(3, 4), name='input_1'),
        >>>     Input(shape=(3, 4), name='input_2'),
        >>> ]
        >>> anon = Anonymous(exp='x[0] + x[1]', name='anon1')
        >>> # This layer just returns the input Tensor as it is
        >>> y = anon(*x)
        >>> # y represents `input_1` + `input_2`

        When input to ``build`` is dict argument, ``x`` in `exp`` represents
        Tensors passed as dict.

        >>> x = {
        >>>     'input_3': Input(shape=(3, 4), name='input_3'),
        >>>     'input_4': Input(shape=(3, 4), name='input_4'),
        >>> }
        >>> anon = Anonymous(exp='x["input_3"] * x["input_4"]', name='anon2')
        >>> # This layer just returns the input Tensor as it is
        >>> y = anon(**x)
        >>> # y represents `input_3` * `input_4`
        """
        self.input = input_tensor
        with variable_scope(self.args['name']):
            self.output = self._build(input_tensor, *args, **kwargs)
            return self.output

    def _build(self, input_tensor, *args, **kwargs):
        # pylint: disable=eval-used
        local = _get_safe_function(input_tensor, *args, **kwargs)
        _LG.info('  Appyling %s on %s', self.args['exp'], local['x'])
        y = eval(self.args['exp'], {'__builtins__': None}, local)
        return wrapper.Tensor(tensor=y.unwrap(), shape=y.shape, name='output')
