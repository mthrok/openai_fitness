"""Module for facilitaging the use of nn module"""
from __future__ import absolute_import
# pylint: disable=wildcard-import
from .io import get_model_config
from .model_maker import *  # noqa

__all__ = ['get_model_config', 'make_io_node', 'make_layer', 'make_model']
