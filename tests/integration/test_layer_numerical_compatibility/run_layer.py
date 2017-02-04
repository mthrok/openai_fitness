from __future__ import absolute_import

import os
import logging

import h5py
import numpy as np

import luchador
import luchador.util
from luchador import nn

_LG = logging.getLogger('luchador')


def _parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Feed batch data to layer and save the output to file'
    )
    ap.add_argument(
        'config',
        help='File contains layer and run config.'
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


def _forward_prop(layer, input_value, parameter_file, n_ite):
    sess = nn.Session()
    input_ = nn.Input(shape=input_value.shape)
    output = layer(input_)
    if parameter_file:
        _LG.info('Loading parameter values from %s', parameter_file)
        sess.load_from_file(parameter_file, strict=False)

    _LG.info('Running forward path for %s times', n_ite)
    for _ in range(n_ite):
        ret = sess.run(
            outputs=output, inputs={input_: input_value.astype(input_.dtype)},
            updates=layer.get_update_operation())
    _LG.info('Run forward path. Output shape: %s', ret.shape)
    return ret


def _transpose_needed(layer, input_shape):
    def _is_convolution():
        return (
            layer.__class__.__name__ == 'Conv2D' and
            luchador.get_nn_backend() == 'tensorflow' and
            luchador.get_nn_conv_format() == 'NHWC'
        )

    def _is_batch_normalization_4d():
        return (
            layer.__class__.__name__ == 'BatchNormalization' and
            len(input_shape) > 2 and
            luchador.get_nn_backend() == 'tensorflow' and
            luchador.get_nn_conv_format() == 'NHWC'
        )
    return _is_convolution() or _is_batch_normalization_4d()


def _run_forward_prop(layer, input_value, parameter_file, iteration=1):
    if _transpose_needed(layer, input_value.shape):
        # All the test data is created floowing the Theano format, which
        # is NCHW for input data. So when running this test in Tensorflow
        # backend, we reorder the input data to NHWC
        input_value_ = input_value.transpose((0, 2, 3, 1))
        _LG.info(
            '  *** Rearranging input shape from %s to %s',
            input_value.shape, input_value_.shape)
        input_value = input_value_

    output = _forward_prop(layer, input_value, parameter_file, iteration)

    if _transpose_needed(layer, input_value.shape):
        # So as to make the output comarison easy, we revert the oreder
        # from NHWC to NCHW.
        output_ = output.transpose((0, 3, 1, 2))
        _LG.info(
            '  *** Rearranging output shape from %s to %s',
            output.shape, output_.shape)
        output = output_

    return output


def _create_layer(typename, args=None):
    return nn.get_layer(typename)(**(args or {}))


def _load_input_value(filepath):
    _LG.info('Loading input value from %s', filepath)
    file_ = h5py.File(filepath, 'r')
    ret = np.asarray(file_['input'])
    file_.close()
    _LG.info('  Shape %s', ret.shape)
    _LG.info('  Dtype %s', ret.dtype)
    return ret


def _save_output(filepath, data):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    _LG.info('Saving output value to %s', filepath)
    _LG.info('  Shape %s', data.shape)
    _LG.info('  Dtype %s', data.dtype)
    file_ = h5py.File(filepath, 'w')
    file_.create_dataset('output', data=data)
    file_.close()


def _load_config(config_path):
    cfg = luchador.util.load_config(config_path)

    cfg_dir = os.path.dirname(config_path)
    input_file = os.path.join(cfg_dir, cfg['input'])
    param_file = (
        os.path.join(cfg_dir, cfg['parameter'])
        if 'parameter' in cfg else None)
    return cfg, input_file, param_file


def _init_logger(debug):
    from luchador.util import initialize_logger
    message_format = (
        '%(asctime)s: %(levelname)5s: %(funcName)10s: %(message)s'
        if debug else '%(asctime)s: %(levelname)5s: %(message)s'
    )
    level = logging.DEBUG if debug else logging.INFO
    initialize_logger(
        name='luchador', message_format=message_format, level=level)


def _main():
    args = _parse_command_line_args()

    _init_logger(args.debug)

    cfg, input_file, param_file = _load_config(args.config)

    output = _run_forward_prop(
        layer=_create_layer(**cfg['layer']),
        input_value=_load_input_value(input_file),
        parameter_file=param_file,
        **cfg.get('run', {})
    )

    if args.output:
        _save_output(args.output, output)


if __name__ == '__main__':
    _main()
