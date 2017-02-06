"""Utility functions related to YAML, used throughout luchador"""
from __future__ import absolute_import

import ruamel.yaml as yaml


__all__ = ['load_config', 'pprint_dict']


def load_config(filepath):
    """Load yaml file"""
    with open(filepath) as file_:
        return yaml.safe_load(file_)


def pprint_dict(dictionary):
    """Pretty-print dictionary in YAML style"""
    return yaml.dump(dictionary, default_flow_style=False)
