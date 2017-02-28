"""Test Model module"""
from __future__ import absolute_import

import ruamel.yaml as yaml

import luchador.nn as nn
from luchador.nn.model import Container
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
      - typename: Dense
        args:
          n_nodes: 5
          name: seq1/layer1/dense

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
      - typename: Dense
        args:
          n_nodes: 6
          name: seq2/layer1/dense

con_1:
  typename: Container
  args:
    input_config:
      typename: Input
      args:
        name: input_seq_3
        shape:
          - null
          - 8
    model_configs:
      - <<: *seq_1
        name: seq_1
    output_config:
      typename: Tensor
      name: seq1/layer1/dense/output
"""

_MODELS = yaml.round_trip_load(_MODEL_DEFS)


class TestContainer(fixture.TestCase):
    """Test Container class"""
    def test_fetch_sequences(self):
        """Container can fetch variables correctly"""
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
        self.assertEqual(
            container.get_update_operations(),
            (
                models[0].get_update_operations() +
                models[1].get_update_operations()
            )
        )

    def test_nested_container(self):
        """Nested Container can fetch variables correctly"""
        models, container = _get_models(
            self.get_scope(),
            [_MODELS['con_1']],
        )
        model = models[0].models['seq_1']
        self.assertEqual(
            container.get_parameter_variables(),
            model.get_parameter_variables(),
        )
        self.assertEqual(
            container.get_output_tensors(),
            model.get_output_tensors(),
        )
        self.assertEqual(
            container.get_update_operations(),
            model.get_update_operations(),
        )
