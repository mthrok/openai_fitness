from __future__ import absolute_import

import unittest

import tensorflow as tf

import luchador
from luchador.nn.core.tensorflow.wrapper import Input
from luchador.nn.core.tensorflow.layer import Dense, Conv2D
from luchador.nn.core.tensorflow.layer import _map_padding
import luchador.nn.core.tensorflow.scope as scp


@unittest.skipUnless(luchador.get_nn_backend() == 'tensorflow',
                     'Tensorflow backend')
class TestConv2D(unittest.TestCase):
    longMessage = True

    def test_map_padding(self):
        """padding string is correctly mapped to valid one"""
        inputs = ('full', 'half', 'same', 'valid')
        expecteds = ('VALID', 'SAME', 'SAME', 'VALID')
        for i, expected in zip(inputs, expecteds):
            found = _map_padding(i)
            self.assertEqual(expected, found)


@unittest.skipUnless(luchador.get_nn_backend() == 'tensorflow',
                     'Tensorflow backend')
class TestDense(unittest.TestCase):
    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        tf.reset_default_graph()
        self.session.close()
        self.session = None

    def test_recreate_success_with_reuse(self):
        """Copied layer can create node when reuse=True in variable scope"""
        n_nodes = 256
        input = Input(shape=(None, n_nodes), name='foo')()
        with scp.variable_scope('test_recreate_with_reuse', reuse=False):
            dense1 = Dense(n_nodes)
            dense1(input)
            with scp.variable_scope(tf.get_variable_scope(), reuse=True):
                dense2 = Dense(**dense1.serialize()['args'])
                dense2(input)

        vars1 = dense1.parameter_variables
        vars2 = dense2.parameter_variables
        expected = vars1.keys()
        found = vars2.keys()
        self.assertEqual(
            expected, found,
            'Copied layer should have the same parameter variables '
            'as the original layer. Expected: {}, Found: {}'
            .format(expected, found))
        for key in vars1.keys():
            expected, found = vars1[key], vars2[key]
            self.assertTrue(
                vars1[key].get() is vars2[key].get(),
                'Variable objects in copied layer should be identical '
                'to those in the original layer. Variable {} is not identical.'
                .format(key)
            )

    def test_copy_fail_without_reuse(self):
        """Copied layer fails to create node when reuse is not True"""
        fmt = luchador.get_nn_conv_format()
        shape = (None, 4, 84, 84) if fmt == 'NCHW' else (None, 84, 84, 4)
        input = Input(shape=shape, name='foo')()
        with scp.variable_scope('test_recreate_without_reuse', reuse=False):
            conv = Conv2D(84, 84, 4, 4)
            conv(input)
            try:
                conv2 = Conv2D(**conv.serialize()['args'])
                conv2(input)
                self.fail('Copied layer should raise ValueError when '
                          'reuse is not enabled in variable scope.')
            except ValueError:
                pass
            except Exception as e:
                self.fail(
                    'Expected ValueError when copied layer tries to '
                    'create node without reuse enabled in variable scope. '
                    'Found "{}"'.format(e)
                )
