"""Test Sequetial model's equality"""
from __future__ import absolute_import

import unittest

from luchador import nn

# pylint: disable=invalid-name


def _make_layers():
    conv = nn.layer.Conv2D(
        filter_height=8, filter_width=8, n_filters=8, strides=4)
    dense1 = nn.layer.Dense(n_nodes=512)
    dense2 = nn.layer.Dense(n_nodes=256)
    return conv, dense1, dense2


class LayerConfigEqualityTest(unittest.TestCase):
    """Test equality of LayerConfig"""
    longMessage = True

    def test_LayerConfig_equality(self):
        """LayerConfig with the same scope and layer should equal"""
        conv, _, _ = _make_layers()
        cfg1 = nn.model.sequential.LayerConfig(layer=conv, scope='foo')
        cfg2 = nn.model.sequential.LayerConfig(layer=conv, scope='foo')
        self.assertEqual(cfg1, cfg2)

    def test_LayerConfig_scope_inequality(self):
        """LayerConfig with different scopes should not equal"""
        conv, _, _ = _make_layers()
        cfg1 = nn.model.sequential.LayerConfig(layer=conv, scope='foo')
        cfg2 = nn.model.sequential.LayerConfig(layer=conv, scope='bar')
        self.assertNotEqual(cfg1, cfg2)

    def test_LayerConfig_layer_inequality(self):
        """LayerConfig with different layers should not equal"""
        _, dense1, dense2 = _make_layers()
        cfg1 = nn.model.sequential.LayerConfig(layer=dense1, scope='foo')
        cfg2 = nn.model.sequential.LayerConfig(layer=dense2, scope='foo')
        self.assertNotEqual(cfg1, cfg2)

    def test_LayerConfig_layer_scope_inequality(self):
        """LayerConfig with different layers and scopes should not equal"""
        _, dense1, dense2 = _make_layers()
        cfg1 = nn.model.sequential.LayerConfig(layer=dense1, scope='foo')
        cfg2 = nn.model.sequential.LayerConfig(layer=dense2, scope='bar')
        self.assertNotEqual(cfg1, cfg2)


class SequentialModelEqualityTest(unittest.TestCase):
    """Test equality of LayerConfig"""
    longMessage = True

    def test_sequential_model_equality_identical_layers(self):
        """Sequential Models with the identical layers should be euqal"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()

        conv, dense, _ = _make_layers()

        m1.add_layer(conv, scope='conv')
        m2.add_layer(conv, scope='conv')

        m1.add_layer(dense, scope='dense')
        m2.add_layer(dense, scope='dense')

        self.assertEqual(m1, m2, 'Models with the identical layers must equal')
        self.assertEqual(m2, m1, 'Models with the identical layers must equal')

    def test_sequential_model_equality_equal_layers(self):
        """Sequential Models with the same layers should be euqal"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()

        conv1, dense1, _ = _make_layers()
        conv2, dense2, _ = _make_layers()

        m1.add_layer(conv1, scope='conv')
        m2.add_layer(conv2, scope='conv')

        m1.add_layer(dense1, scope='dense')
        m2.add_layer(dense2, scope='dense')

        self.assertEqual(m1, m2, 'Models with the same layers must equal')
        self.assertEqual(m2, m1, 'Models with the same layers must equal')

    def test_model_equality_empty(self):
        """Models without layers should be euqal"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()
        self.assertEqual(m1, m2, 'Models without layers must be equal')

    def test_model_self_equality(self):
        """Sequential model always equal to self"""
        m = nn.model.sequential.Sequential()
        self.assertEqual(m, m)

        conv, dense1, dense2 = _make_layers()
        m.add_layer(conv, scope='conv')
        self.assertEqual(m, m)

        m.add_layer(dense1, scope='dense')
        self.assertEqual(m, m)

        m.add_layer(dense2, scope='dense')
        self.assertEqual(m, m)

    def test_model_inequality_type(self):
        """Sequential model do not equal to other types"""
        m = nn.model.sequential.Sequential()

        conv, dense, _ = _make_layers()

        m.add_layer(conv, scope='conv')
        m.add_layer(dense, scope='dense')

        self.assertNotEqual(
            m, 1, 'A model instance should not equal to an int')
        self.assertNotEqual(
            m, {}, 'A model instance should not equal to a dict')
        self.assertNotEqual(
            m, [], 'A model instance should not equal to a list')
        self.assertNotEqual(
            m, '', 'A model instance should not equal to a str')
        self.assertNotEqual(
            m, object(), 'A model instance should not be equal with an object')

    def test_model_inequality_number_of_layers(self):
        """Sequential models with different #layers should not equal"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()

        conv, dense, _ = _make_layers()

        m1.add_layer(conv, scope='conv')
        m2.add_layer(conv, scope='conv')

        m1.add_layer(dense, scope='dense')

        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m2, m1)

    def test_model_inequality_layer_types(self):
        """Sequential models with different layers should not equal"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()

        conv, dense, _ = _make_layers()

        m1.add_layer(conv, scope='conv')

        m2.add_layer(dense, scope='dense')

        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m2, m1)

    def test_model_inequality_layer_configurations(self):
        """Sequential models with different layers should not equal"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()

        _, dense1, dense2 = _make_layers()

        m1.add_layer(dense1, scope='dense1')

        m2.add_layer(dense2, scope='dense2')

        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m2, m1)

    def test_model_equality_serialized_configurations(self):
        """Configurations serialized from same models are same"""
        m1 = nn.model.sequential.Sequential()
        m2 = nn.model.sequential.Sequential()

        conv, dense1, dense2 = _make_layers()
        m1.add_layer(conv, scope='conv')
        m1.add_layer(dense1, scope='dense1')
        m1.add_layer(dense2, scope='dense2')

        conv, dense1, dense2 = _make_layers()
        m2.add_layer(conv, scope='conv')
        m2.add_layer(dense1, scope='dense1')
        m2.add_layer(dense2, scope='dense2')

        self.assertEqual(m1.serialize(), m2.serialize())
