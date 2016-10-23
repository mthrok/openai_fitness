from __future__ import absolute_import

import yaml


def load_config(filepath):
    with open(filepath) as file_:
        return yaml.load(file_)
