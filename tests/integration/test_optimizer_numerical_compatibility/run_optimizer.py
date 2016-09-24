from __future__ import absolute_import

import os
import csv
import logging

from luchador.util import load_config
from luchador.nn import (
    Session,
    get_optimizer,
)
import formula

_LG = logging.getLogger('luchador')
_LG.setLevel(logging.INFO)


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Run optimizer on simple formular and save the result'
    )
    ap.add_argument(
        'formula',
        help='Name of formula on which optimizer is run.'
    )
    ap.add_argument(
        'optimizer',
        help='Optimizer configuration to test'
    )
    ap.add_argument(
        '--iterations', default=100, type=int,
        help='#Minimize operations to run'
    )
    ap.add_argument(
        '--output',
        help='File path to save result.'
    )
    return ap.parse_args()


def get_formula(name):
    try:
        return getattr(formula, name).get()
    except Exception:
        _LG.exception('Formula ({}) not found.'.format(name))
        raise


def optimize(optimizer, loss, wrt, n_ite):
    minimize_op = optimizer.minimize(loss=loss, wrt=wrt)
    sess = Session()
    sess.initialize()
    result = []
    for _ in range(n_ite+1):
        output = sess.run(outputs=[loss, wrt], name='output')
        result.append({
            'loss': output[0],
            'wrt': output[1],
        })
        sess.run(updates=minimize_op, name='minimize')
    return result


def load_optimizer(filepath):
    cfg = load_config(filepath)
    Optimizer = get_optimizer(cfg['name'])
    return Optimizer(**cfg['args'])


def save_result(filepath, result):
    directory = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # TODO: Limit data points
    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result[0].keys())
        writer.writeheader()
        for r in result:
            writer.writerow(r)


def main():
    args = parse_command_line_args()
    _LG.info('  Running {} on {} for {} times'.format(
        args.optimizer, args.formula, args.iterations))
    formula = get_formula(args.formula)
    optimizer = load_optimizer(args.optimizer)
    result = optimize(
        optimizer=optimizer, loss=formula['loss'],
        wrt=formula['wrt'], n_ite=args.iterations)
    _LG.info('    Y: {} -> {}'.format(
        result[0]['loss'], result[-1]['loss']))
    _LG.info('    X: {} -> {}'.format(
        result[0]['wrt'], result[-1]['wrt']))
    if args.output:
        _LG.info('  Saving at {}'.format(args.output))
        save_result(args.output, result)

if __name__ == '__main__':
    main()
