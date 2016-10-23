from __future__ import absolute_import

import os
import StringIO

import yaml

from luchador.nn import get_model, get_layer

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data')


def make_model(model_config):
    """Make model from model configuration

    Args:
      model_condig (JSON-compatible object): model configuration.

    Returns:
      Model
    """
    model = get_model(model_config['model_type'])()
    for cfg in model_config['layer_configs']:
        scope = cfg['scope']
        layer_cfg = cfg['layer']
        layer = get_layer(layer_cfg['name'])(**layer_cfg['args'])
        model.add_layer(layer=layer, scope=scope)
    return model


def get_model_config(model_name, **parameters):
    """Load pre-defined model configurations

    Args:
      model_name (name): Model name or path to YAML file
      parameters: Parameter for model config

    Returns:
      JSON-compatible object: model configuration.
    """
    file_name = model_name
    if not file_name.endswith('.yml'):
        file_name = '{}.yml'.format(file_name)

    if os.path.isfile(file_name):
        file_path = file_name
    elif os.path.isfile(os.path.join(_DATA_DIR, file_name)):
        file_path = os.path.join(_DATA_DIR, file_name)
    else:
        raise ValueError(
            'No model definition file ({}) found.'.format(file_name))

    with open(file_path, 'r') as f:
        model_text = f.read()

    if parameters:
        model_text = model_text.format(**parameters)

    model_text = StringIO.StringIO(model_text)
    return yaml.load(model_text)
