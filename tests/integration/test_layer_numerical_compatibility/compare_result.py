from __future__ import division

import logging

import h5py
import numpy as np

import luchador  # noqa
_LG = logging.getLogger('luchador')


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Load two files and check if their values are similar'
    )
    ap.add_argument(
        'input1', help='Text file that contains single value series'
    )
    ap.add_argument(
        'input2', help='Text file that contains single value series'
    )
    ap.add_argument(
        '--threshold', type=float, default=1e-2,
        help='Relative threshold for value comparison'
    )
    return ap.parse_args()


def load_result(filepath):
    f = h5py.File(filepath, 'r')
    ret = np.asarray(f['output'])
    f.close()
    return ret


def check(arr1, arr2, abs_threshold=0.00015, relative_threshold=1e-1):
    abs_diff = np.absolute(arr1 - arr2)
    rel_diff = abs_diff / (arr1 + arr2 + 1)
    _LG.info('  Ave absolute diff: {}'.format(abs_diff.mean()))
    _LG.info('  Max absolute diff: {}'.format(abs_diff.max()))
    _LG.info('  Min absolute diff: {}'.format(abs_diff.min()))
    _LG.info('  Ave relative diff: {}'.format(rel_diff.mean()))
    _LG.info('  Max relative diff: {}'.format(rel_diff.max()))
    _LG.info('  Min relative diff: {}'.format(rel_diff.min()))
    return (
            (abs_diff > abs_threshold).any() or
            (rel_diff > relative_threshold).any()
    )


def main():
    args = parse_command_line_args()
    _LG.info('Comparing {} and {}. (Threshold: {} [%])'
             .format(args.input1, args.input2, 100 * args.threshold))
    data1 = load_result(args.input1)
    data2 = load_result(args.input2)

    if check(data1, data2, relative_threshold=args.threshold):
        raise ValueError('Data are different')
    _LG.info('Okay')

if __name__ == '__main__':
    main()
