from __future__ import absolute_import

import logging

import h5py
import numpy as np

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
    return ap.parse_args()


def forward_prop(layer, input_value, parameter_file=None):
    sess = Session()
    if parameter_file:
        _LG.info('Loading {}'.format(parameter_file))
        sess.load_from_file(parameter_file)
    input = Input(shape=input_value.shape,
                  dtype=input_value.dtype)
    output = layer(input.build())
    return sess.run(outputs=output, inputs={input: input_value})


def load_layer(filepath):
    cfg = load_config(filepath)
    Layer = get_layer(cfg['name'])
    return Layer(**cfg['args'])


def load_input_value(filepath):
    _LG.info('  Loading {}'.format(filepath))
    f = h5py.File(filepath, 'r')
    ret = np.asarray(f['input'])
    f.close()
    _LG.info('    Shape {}'.format(ret.shape))
    _LG.info('    Dtype {}'.format(ret.dtype))
    return ret


def save_result(filepath, data):
    _LG.info('  Saving  {}'.format(filepath))
    _LG.info('    Shape {}'.format(data.shape))
    _LG.info('    Dtype {}'.format(data.dtype))
    f = h5py.File(filepath, 'w')
    f.create_dataset('result', data=data)
    f.close()


def main():
    args = parse_command_line_args()
    result = forward_prop(layer=load_layer(args.layer),
                          input_value=load_input_value(args.input),
                          parameter_file=args.parameter)
    if args.output:
        save_result(args.output, result)

if __name__ == '__main__':
    main()
