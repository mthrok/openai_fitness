"""Defined Network model architecture and utility tools for build"""
from __future__ import absolute_import

from . import model  # noqa: F401

from .model import get_model  # noqa: F401
from .util import (  # noqa: F401
    make_model,
    get_model_config,
)
