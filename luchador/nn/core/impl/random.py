"""Implement wrapper classes"""
from __future__ import absolute_import

from ..backend import random

# pylint: disable=invalid-name

NormalRandom = random.NormalRandom
UniformRandom = random.UniformRandom

__all__ = ['NormalRandom', 'UniformRandom']
