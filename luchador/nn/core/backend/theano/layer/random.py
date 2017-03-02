"""Module for implementing random source"""
from __future__ import absolute_import

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..wrapper import Tensor
# pylint: disable=no-member


class _NoiseMixin(object):  # pylint: disable=too-few-public-methods
    def _instantiate_generator(self):
        if self._rng is None:
            seed = self.args['seed'] or 123456
            self._rng = RandomStreams(seed=seed)

    def _build(self, input_tensor):
        self._instantiate_generator()

        shape = input_tensor.shape
        dtype = input_tensor.dtype
        noise = self._sample(shape=shape, dtype=dtype)

        if self.args['mode'] == 'add':
            tensor = input_tensor.unwrap() + noise
        else:
            tensor = input_tensor.unwrap() * noise
        return Tensor(tensor=tensor, shape=shape, name='output')


class NormalNoise(_NoiseMixin):
    """Implement NormalNoise in theano backend.

    See :any:`BaseNormalNoise` for the detail
    """
    def _sample(self, shape, dtype):
        mean, std = self.args['mean'], self.args['std']
        return self._rng.normal(
            size=shape, avg=mean, std=std, dtype=dtype,
        )


class UniformNoise(_NoiseMixin):
    """Implement UniformNoise in theano backend.

    See :any:`BaseUniformNoise` for the detail
    """
    def _sample(self, shape, dtype):
        low, high = self.args['low'], self.args['high']
        return self._rng.uniform(
            size=shape, low=low, high=high, dtype=dtype,
        )
