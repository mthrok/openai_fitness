"""Test nn.model.util module"""
from __future__ import absolute_import

import luchador
from luchador import nn

from tests.unit import fixture


class UtilTest(fixture.TestCase):
    """Test model [de]serialization"""
    longMessage = True
    maxDiff = None

    def test_create_model(self):
        """Deserialized model is equal to the original"""
        fmt = luchador.get_nn_conv_format()
        shape = '[null, 4, 84, 84]' if fmt == 'NCHW' else '[null, 84, 84, 4]'
        cfg1 = nn.get_model_config(
            'vanilla_dqn', n_actions=5, input_shape=shape)

        with nn.variable_scope(self.get_scope()):
            m1 = nn.make_model(cfg1)
            m2 = nn.make_model(m1.serialize())
            self.assertEqual(m1, m2)

    def test_input_tensor(self):
        """Tensor type input correctly build model on existing TEnsor"""
        model_def1 = {
            'model_type': 'Sequential',
            'input': {
                'typename': 'Input',
                'args': {
                    'shape': [None, 3]
                },
                'name': 'input'
            },
            'layer_configs': [{
                'scope': 'layer1/dense',
                'typename': 'Dense',
                'args': {
                    'n_nodes': 4,
                }
            }]
        }

        model_def2 = {
            'model_type': 'Sequential',
            'input': {
                'typename': 'Tensor',
                'name': '{}/layer1/dense/output'.format(self.get_scope()),
            },
            'layer_configs': [{
                'scope': 'layer1/ReLU',
                'typename': 'ReLU',
            }]
        }

        with nn.variable_scope(self.get_scope()):
            model1 = nn.make_model(model_def1)

            model2 = nn.make_model(model_def2)

            self.assertIs(model1.output, model2.input)
