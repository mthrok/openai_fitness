import unittest

import theano
import numpy as np

import luchador

# import theano
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'


@unittest.skipUnless(luchador.get_nn_backend() == 'theano', 'Theano backend')
class TestConv2D(unittest.TestCase):
    longMessage = True

    def setUp(self):
        from luchador.nn import scope
        scope._reset()

    def test_map_border_mode(self):
        """padding string is correctly mapped to border_mode"""
        from luchador.nn.core.theano import layer
        inputs = ('full', 'half', 'SAME', 'valid')
        expecteds = ('full', 'half', 'half', 'valid')
        for i, expected in zip(inputs, expecteds):
            found = layer._map_border_mode(i)
            self.assertEqual(expected, found)

    def _test_shape_inference(
            self, input_shape, filter_shape, n_filters, strides, padding):
        from luchador import nn
        from luchador.nn.core.theano import scope
        scope._reset()
        conv2d = nn.Conv2D(filter_height=filter_shape['height'],
                           filter_width=filter_shape['width'],
                           n_filters=n_filters,
                           strides=strides, padding=padding)
        input_ = nn.Input(shape=input_shape)()
        output = conv2d(input_)

        f = theano.function([input_.unwrap()], output.unwrap())
        input_value = np.zeros(input_shape)
        output_value = f(input_value)

        expected = output_value.shape
        found = output.shape
        self.assertEqual(
            found, expected,
            'Shape inference failed. Expected: {}, Found: {}. '
            'input_shape: {}, filter_shape: {}, n_filters: {}, '
            'strides: {}, padding: {}'
            .format(expected, found, input_shape, filter_shape,
                    n_filters, strides, padding))

    def test_shape_inference_odd_valid(self):
        """output_shape is correct when filter+image:odd, mode:valid"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 6, 'width': 9}
        n_filters = 4
        padding = 'valid'
        strides = (1, 1)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
        strides = (2, 1)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
        strides = (1, 2)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_odd_half(self):
        """output_shape is correct when filter+image:odd and mode:half"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 6, 'width': 9}
        n_filters = 4
        padding = 'half'

        strides = (1, 1)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
        strides = (2, 1)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
        strides = (1, 2)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_odd_full(self):
        """output_shape is correct when filter+image:odd and mode:full"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 6, 'width': 9}
        n_filters = 4
        padding = 'full'

        strides = (1, 1)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
        strides = (2, 1)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
        strides = (1, 2)
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_valid_1(self):
        """output_shape is correct when filter+image:even, mode:valid, pad=1"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'valid'
        strides = 1
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_valid_2(self):
        """output_shape is correct when filter+image:even, mode:valid, pad=2"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'valid'
        strides = 2
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_valid_3(self):
        """output_shape is correct when filter+image:even, mode:valid, pad=3"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'valid'
        strides = 3
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_valid_5(self):
        """output_shape is correct when filter+image:even, mode:valid, pad=5"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'valid'
        strides = 5
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_full_1(self):
        """output_shape is correct when filter+image:even, mode:full, pad=1"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'full'
        strides = 1
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_full_2(self):
        """output_shape is correct when filter+image:even, mode:full, pad=2"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'full'
        strides = 2
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_full_3(self):
        """output_shape is correct when filter+image:even, mode:full, pad=3"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'full'
        strides = 3
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_full_5(self):
        """output_shape is correct when filter+image:even, mode:full, pad=5"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'full'
        strides = 5
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_half_1(self):
        """output_shape is correct when filter+image:even, mode:half, pad=1"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'half'
        strides = 1
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_half_2(self):
        """output_shape is correct when filter+image:even, mode:half, pad=2"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'half'
        strides = 2
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_half_3(self):
        """output_shape is correct when filter+image:even, mode:half, pad=3"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'half'
        strides = 3
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)

    def test_shape_inference_even_half_5(self):
        """output_shape is correct when filter+image:even, mode:half, pad=5"""
        input_shape = (32, 4, 73, 84)
        filter_shape = {'height': 5, 'width': 8}
        n_filters = 4
        padding = 'half'
        strides = 5
        self._test_shape_inference(
            input_shape, filter_shape, n_filters, strides, padding)
