from __future__ import division

import csv
import logging

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


def load_data(filepath):
    with open(filepath, 'r') as csvfile:
        loss, wrt = [], []
        for row in csv.DictReader(csvfile):
            loss.append(float(row['loss']))
            wrt.append(float(row['wrt']))
        return {'loss': loss, 'wrt': wrt}


def check(series1, series2, abs_threshold=0.00015, relative_threshold=1e-1):
    res = []
    for i, (val1, val2) in enumerate(zip(series1[1:], series2[1:])):
        abs_diff = abs(val1 - val2)
        rel_diff = abs_diff / (val1 + val2)
        if (
                abs_diff > abs_threshold and
                rel_diff > relative_threshold
        ):
            res.append((i, val1, val2))
    return res


def main():
    args = parse_command_line_args()
    _LG.info('Comparing {} and {}. (Threshold: {} [%])'
             .format(args.input1, args.input2, 100 * args.threshold))
    data1 = load_data(args.input1)
    data2 = load_data(args.input2)

    message = ''
    res = check(data1['loss'], data2['loss'],
                relative_threshold=args.threshold)
    error_ratio = len(res) / len(data1['loss'])
    if res:
        message += 'Loss are different\n'
        for i, val1, val2 in res:
            message += 'Line {}: {}, {}\n'.format(i, val1, val2)

    res = check(data1['wrt'], data2['wrt'],
                relative_threshold=args.threshold)
    if res:
        message += 'wrt are different\n'
        for i, val1, val2 in res:
            message += 'Line {}: {}, {}\n'.format(i, val1, val2)

    if message:
        _LG.error(message)
        raise ValueError(
            'Data are different at {} % points'
            .format(100 * error_ratio)
        )
    else:
        _LG.info('Okay')

if __name__ == '__main__':
    main()
