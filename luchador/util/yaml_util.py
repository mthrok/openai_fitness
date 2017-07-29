"""Utility functions related to YAML, used throughout luchador"""
from __future__ import absolute_import

from six import StringIO
import ruamel.yaml as yaml

from .misc import is_iteratable

__all__ = ['load_config', 'pprint_dict']


def _convert_to_str(value):
    """Convert object into one-line YAML string"""
    if isinstance(value, str):
        return value
    if value is None:
        return 'null'

    if isinstance(value, dict):
        return '{{{}}}'.format(
            ', '.join(
                '{}: {}'.format(key, _convert_to_str(val))
                for key, val in value.items()
            )
        )

    if is_iteratable(value):
        return '[{}]'.format(
            ', '.join([_convert_to_str(val) for val in value]))
    return str(value)


def load_config(filepath, **parameters):
    """Load YAML file and dynamically update values.

    Parameters
    ----------
    filepath : str
        File path

    **parameters
        Additional parameter to provide missing value.

    Example
    -------
    When following YAML file is given as ``test.yml``,

    .. code-block:: YAML

        model_type: Sequential
          - scope: layer1
            typename: Dense
            args:
              n_nodes: {n_actions}

    by loading the file as ``load_config('test.yml', n_actions=5)``,
    ``{n_actions}`` is overwritten with ``5``.
    """
    with open(filepath, 'r') as file_:
        model_text = file_.read()

    if parameters:
        model_text = model_text.format(**{
            key: _convert_to_str(val)
            for key, val in parameters.items()
        })

    model_text = StringIO(model_text)
    return yaml.safe_load(model_text)


def pprint_dict(dictionary):
    """Pretty-print dictionary in YAML style"""
    return yaml.dump(
        dictionary, default_flow_style=False, Dumper=yaml.RoundTripDumper)
