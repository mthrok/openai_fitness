"""Test Model module"""
from __future__ import absolute_import

import ruamel.yaml as yaml

import luchador.util
import luchador.nn as nn

from luchador.nn.model import Sequential, Container

from tests.unit import fixture


def _make_model(scope, model_config):
    with nn.variable_scope(scope):
        return nn.util.make_model(model_config)


def _get_models(scope, model_configs):
    models, container = [], Container()
    for i, cfg in enumerate(model_configs):
        model = _make_model(scope, cfg)
        models.append(model)
        container.add_model('model_{}'.format(i), model)
    return models, container


_MODEL_DEFS = """
seq_1: &seq_1
  typename: Sequential
  args:
    input_config:
      typename: Input
      args:
        name: input_seq_1
        shape:
          - null
          - 4
    layer_configs:
      - scope: seq1/layer1/dense
        typename: Dense
        args:
          n_nodes: 5

seq_2:
  typename: Sequential
  args:
    input_config:
      typename: Input
      args:
        name: input_seq_2
        shape:
          - null
          - 5
    layer_configs:
      - scope: seq2/layer1/dense
        typename: Dense
        args:
          n_nodes: 6

con_1:
  typename: Container
  args:
    input_config:
      typename: Input
      args:
        name: input_seq_3,
        shape:
          - null
          - 8
    model_configs:
      <<: *seq_1
      name: seq_1
"""

_MODELS = yaml.round_trip_load(_MODEL_DEFS)


class TestContainer(fixture.TestCase):
    """Test Container class"""
    def test_fetch_sequences(self):
        """Fetch parameters from all the models."""
        models, container = _get_models(
            self.get_scope(),
            [_MODELS['seq_1'], _MODELS['seq_2']],
        )

        self.assertEqual(
            container.get_parameter_variables(),
            (
                models[0].get_parameter_variables() +
                models[1].get_parameter_variables()
            )
        )
        self.assertEqual(
            container.get_output_tensors(),
            (
                models[0].get_output_tensors() +
                models[1].get_output_tensors()
            )
        )
        # TODO: Revise
        '''
        self.assertEqual(
            container.get_update_operations(),
            (
                models[0].get_update_operations() +
                models[1].get_update_operations()
            )
        )
        '''
        # def test_nested_
