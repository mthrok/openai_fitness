"""Compare results of layer IO test over backends"""
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np


def _parse_command_line_args():
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


def _load_result(filepath):
    file_ = h5py.File(filepath, 'r')
    ret = np.asarray(file_['output'])
    file_.close()
    return ret


def _print_stats(*arrs):
    print('{sum:>10}  {max:>10}  {min:>10}  {mean:>10}'
          .format(sum='sum', max='max', min='min', mean='mean'))
    for arr in arrs:
        sum_, max_, min_, mean = arr.sum(), arr.max(), arr.min(), arr.mean()
        print('{sum:10.3E}  {max:10.3E}  {min:10.3E}  {mean:10.3E}'
              .format(sum=sum_, max=max_, min=min_, mean=mean))
    print('')


def _check(arr1, arr2, abs_threshold=0.00015, relative_threshold=1e-1):
    _print_stats(arr1, arr2)
    abs_diff = np.absolute(arr1 - arr2)
    rel_diff = abs(abs_diff / (arr1 + arr2 + 1))
    print('  Ave absolute diff: {}'.format(abs_diff.mean()))
    print('  Max absolute diff: {}'.format(abs_diff.max()))
    print('  Min absolute diff: {}'.format(abs_diff.min()))
    print('  Ave relative diff: {}'.format(rel_diff.mean()))
    print('  Max relative diff: {}'.format(rel_diff.max()))
    print('  Min relative diff: {}'.format(rel_diff.min()))
    return (
        (abs_diff > abs_threshold).any() or
        (rel_diff > relative_threshold).any()
    )


def _main():
    args = _parse_command_line_args()
    print('Comparing {} and {}. (Threshold: {} [%])'
          .format(args.input1, args.input2, 100 * args.threshold))
    data1 = _load_result(args.input1)
    data2 = _load_result(args.input2)

    if _check(data1, data2, relative_threshold=args.threshold):
        raise ValueError('Data are different')
    print('Okay')


if __name__ == '__main__':
    _main()
