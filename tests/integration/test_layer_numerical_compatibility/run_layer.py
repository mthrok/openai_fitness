from __future__ import absolute_import

import os
import logging

import h5py
import numpy as np

import luchador
import luchador.util
from luchador import nn

_LG = logging.getLogger('luchador')

_BE = luchador.get_nn_backend()
_CONV = luchador.get_nn_conv_format()


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


def _run_forward_prop(layer, input_value, parameter_file, iteration=1):
    sess = nn.Session()
    if parameter_file:
        _LG.info('Loading parameter values from %s', parameter_file)
        sess.load_from_file(parameter_file, strict=False)

    _LG.info('Running forward path for %s times', iteration)
    for _ in range(iteration):
        ret = sess.run(
            outputs=layer.output, updates=layer.get_update_operations(),
            inputs={layer.input: input_value.astype(layer.input.dtype)},
            name='test',
        )
    _LG.info('Run forward path. Output shape: %s', ret.shape)
    return ret


def _transpose_needed(input_ndim):
    return input_ndim == 4 and _BE == 'tensorflow' and _CONV == 'NHWC'


def _load_input_value(filepath):
    _LG.info('Loading input value from %s', filepath)
    file_ = h5py.File(filepath, 'r')
    value = np.asarray(file_['input'])
    file_.close()

    _LG.info('  Shape %s', value.shape)
    _LG.info('  Dtype %s', value.dtype)
    return value


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


def _make_model(config_path, input_shape):
    config = luchador.nn.get_model_config(config_path, input_shape=input_shape)
    model_def = config['model']
    return nn.make_model(model_def)


def _main():
    args = _parse_command_line_args()
    _init_logger(args.debug)

    cfg, input_file, param_file = _load_config(args.config)
    input_value = _load_input_value(input_file)

    if _transpose_needed(input_value.ndim):
        # All the test data is created floowing the Theano format, which
        # is NCHW for input data. So when running this test in Tensorflow
        # backend, we reorder the input data to NHWC
        _LG.info('  * Converting input value from NCHW -> NHWC')
        input_value = input_value.transpose((0, 2, 3, 1))

    model = _make_model(args.config, input_shape=input_value.shape)

    output = _run_forward_prop(
        layer=model,
        input_value=input_value,
        parameter_file=param_file,
        **cfg.get('run', {})
    )

    if _transpose_needed(input_value.ndim):
        # So as to make the output comarison easy, we revert the oreder
        # from NHWC to NCHW.
        output_ = output.transpose((0, 3, 1, 2))
        _LG.info(
            '  * Rearranging output shape from %s to %s',
            output.shape, output_.shape)
        output = output_

    if args.output:
        _save_output(args.output, output)


if __name__ == '__main__':
    _main()
