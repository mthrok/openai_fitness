from __future__ import absolute_import

import logging

import h5py
import numpy as np

from luchador import get_nn_backend, get_nn_conv_format
from luchador.util import load_config
from luchador.nn import (
    Input,
    Session,
    get_layer,
)


_LG = logging.getLogger('luchador')
_LG.setLevel(logging.INFO)


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Feed batch data to layer and save the output to file'
    )
    ap.add_argument(
        'layer',
        help='Layer configuration file.'
    )
    ap.add_argument(
        'input',
        help='Input data file. Must be HDF5 data with dataset named "input"'
    )
    ap.add_argument(
        '--parameter',
        help='Layer paramter file.'
    )
    ap.add_argument(
        '--output',
        help='Output data file.'
    )
    ap.add_argument(
        '--debug',
        help='Enable debug log', action='store_true'
    )
    return ap.parse_args()


def forward_prop(layer, input_value, parameter_file):
    sess = Session()
    input = Input(shape=input_value.shape, dtype=input_value.dtype)
    output = layer(input.build())
    if parameter_file:
        _LG.info('Loading parameter values from {}'.format(parameter_file))
        sess.load_from_file(parameter_file)
    ret = sess.run(outputs=output, inputs={input: input_value})
    _LG.info('Run forward path. Output shape: {}'.format(ret.shape))
    return ret


def transpose_needed(layer):
    return (
        layer.__class__.__name__ == 'Conv2D' and
        get_nn_backend() == 'tensorflow' and
        get_nn_conv_format() == 'NHWC'
    )


def run_forward_prop(layer, input_value, parameter_file):
    if transpose_needed(layer):
        # All the test data is created floowing the Theano format, which
        # is NCHW for input data. So when running this test in Tensorflow
        # backend, we reorder the input data to NHWC
        input_value_ = input_value.transpose((0, 2, 3, 1))
        _LG.info(
            '  *** Rearranging input shape from {} to {}'
            .format(input_value.shape, input_value_.shape))
        input_value = input_value_

    output = forward_prop(layer, input_value, parameter_file)

    if transpose_needed(layer):
        # So as to make the output comarison easy, we revert the oreder
        # from NHWC to NCHW.
        output_ = output.transpose((0, 3, 1, 2))
        _LG.info('  *** Rearranging output shape from {} to {}'
                 .format(output.shape, output_.shape))
        output = output_

    return output


def load_layer(filepath):
    cfg = load_config(filepath)
    Layer = get_layer(cfg['name'])
    return Layer(**cfg['args'])


def load_input_value(filepath):
    _LG.info('Loading input value from {}'.format(filepath))
    f = h5py.File(filepath, 'r')
    ret = np.asarray(f['input'])
    f.close()
    _LG.info('  Shape {}'.format(ret.shape))
    _LG.info('  Dtype {}'.format(ret.dtype))
    return ret


def save_output(filepath, data):
    _LG.info('Saving output value to {}'.format(filepath))
    _LG.info('  Shape {}'.format(data.shape))
    _LG.info('  Dtype {}'.format(data.dtype))
    f = h5py.File(filepath, 'w')
    f.create_dataset('output', data=data)
    f.close()


def main():
    args = parse_command_line_args()

    if args.debug:
        _LG.setLevel(logging.DEBUG)

    output = run_forward_prop(
        layer=load_layer(args.layer),
        input_value=load_input_value(args.input),
        parameter_file=args.parameter
    )

    if args.output:
        save_output(args.output, output)

if __name__ == '__main__':
    main()
