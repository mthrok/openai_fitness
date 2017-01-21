from __future__ import absolute_import

import unittest

from luchador import nn

from tests.unit.fixture import get_all_layers

# pylint: disable=invalid-name

LAYERS = get_all_layers()
N_LAYERS = 11

PARAMETERIZED_LAYER_CLASSES = (
    nn.layer.Dense,
    nn.layer.Conv2D,
    nn.layer.TrueDiv,
    nn.layer.BatchNormalization,
)

FIXED_LAYER_CLASSES = (
    nn.layer.Flatten,
    nn.layer.ReLU,
    nn.layer.Sigmoid,
    nn.layer.Softmax,
    nn.layer.NCHW2NHWC,
    nn.layer.NHWC2NCHW,
)


ARGS1 = {
    'Dense': {
        'n_nodes': 100,
    },
    'Conv2D': {
        'filter_height': 8,
        'filter_width': 8,
        'n_filters': 32,
        'strides': 1,
        'padding': 'VALID',
    },
    'TrueDiv': {
        'denom': 1,
    },
    'Concat': {
        'var_list': [
            ('scope1', 'name1'),
            ('scope2', 'name2'),
        ],
        'axis': 1,
    },
    'BatchNormalization': {
        'offset': 0.0,
        'scale': 1.0,
    },
}

ARGS2 = {
    'Dense': {
        'n_nodes': 200,
    },
    'Conv2D': {
        'filter_height': 4,
        'filter_width': 4,
        'n_filters': 16,
        'strides': 2,
        'padding': 'SAME',
    },
    'TrueDiv': {
        'denom': 255
    },
    'Concat': {
        'var_list': [
            ('scope1', 'name1'),
            ('scope2', 'name2'),
        ],
        'axis': 2,
    },
    'BatchNormalization': {
        'offset': 0.5,
        'scale': 1.0,
    },
}


def _make_parameterized_layers(args):
    return [Layer(**args[Layer.__name__])
            for Layer in PARAMETERIZED_LAYER_CLASSES]


def _make_fixed_layers():
    return [Layer() for Layer in FIXED_LAYER_CLASSES]


class LayerTest(unittest.TestCase):
    longMessage = True

    def test_layer_test_coverage(self):
        """All initializers are tested"""
        self.assertEqual(
            len(LAYERS), N_LAYERS,
            'Number of layers are changed. (New layer is added?) '
            'Fix unittest to cover new layers'
        )

    def test_get_layer(self):
        """get_layer returns correct layer class"""
        for name, expected in get_all_layers().items():
            found = nn.get_layer(name)
            self.assertEqual(
                expected, found,
                'get_layer returned wrong layer Class. '
                'Expected: {}, Found: {}.'.format(expected, found)
            )

    def test_parameterized_layer_equality(self):
        """Parameterized layers with same arguments must be equal"""
        layers1 = _make_parameterized_layers(ARGS1)
        layers2 = _make_parameterized_layers(ARGS1)
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(
                l1, l2,
                '\n{} layers initialized with the same arguments '
                'must be equal.'.format(type(l1)))

    def test_parameterized_layer_inequality(self):
        """Parameterized layers with different arguments must not be equal"""
        layers1 = _make_parameterized_layers(ARGS1)
        layers2 = _make_parameterized_layers(ARGS2)
        for l1, l2 in zip(layers1, layers2):
            self.assertNotEqual(
                l1, l2,
                '\n{} layers initialized with the differnet arguments '
                'must not be equal.'.format(type(l1)))

    def test_fixed_layer_equality(self):
        """Instances of not parmaeterized layers must be equal"""
        layers1 = _make_fixed_layers()
        layers2 = _make_fixed_layers()
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(
                l1, l2, '\n{} layers must be eqauel'.format(type(l1)))

    def test_layer_serialize(self):
        """`serialize` method should return constructor arguments """
        layers = _make_parameterized_layers(ARGS1)
        for layer in layers:
            args = layer.serialize()
            layer_name = args['name']
            given_args = ARGS1[layer_name]
            for key, expected in given_args.items():
                found = args['args'][key]
                self.assertEqual(
                    expected, found,
                    '\nArgument {} serialized from {} are different from '
                    'what was given in constructor.'
                    .format(key, layer_name))

    def test_layer_equality_serialize(self):
        """Layers recreated with serialized arguments equal to originals"""
        for layer0 in _make_parameterized_layers(ARGS1):
            args = layer0.serialize()
            layer1 = nn.get_layer(args['name'])(**args['args'])
            expected = layer0
            found = layer1
            self.assertEqual(
                expected, found,
                '\nLayer recreated from serialize arguments must '
                'equal to the original.')
