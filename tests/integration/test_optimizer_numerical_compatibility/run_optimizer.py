"""Run optimizers and save numerical results"""
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


def _parse_command_line_args():
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


def _get_formula(name):
    try:
        return getattr(formula, name).get()
    except Exception:
        _LG.exception('Formula (%s) not found.', name)
        raise


def _optimize(optimizer, loss, wrt, n_ite):
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


def _load_optimizer(filepath):
    cfg = load_config(filepath)
    return get_optimizer(cfg['name'])(**cfg.get('args', {}))


def _save_result(filepath, result):
    directory = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # TODO: Limit data points
    with open(filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result[0].keys())
        writer.writeheader()
        for res in result:
            writer.writerow(res)


def _main():
    args = _parse_command_line_args()
    _LG.info(
        '  Running %s on %s for %s times',
        args.optimizer, args.formula, args.iterations)
    formula_ = _get_formula(args.formula)
    optimizer = _load_optimizer(args.optimizer)
    result = _optimize(
        optimizer=optimizer, loss=formula_['loss'],
        wrt=formula_['wrt'], n_ite=args.iterations)
    _LG.info('    Y: %f -> %f', result[0]['loss'], result[-1]['loss'])
    _LG.info('    X: %f -> %f', result[0]['wrt'], result[-1]['wrt'])
    if args.output:
        _LG.info('  Saving at %s', args.output)
        _save_result(args.output, result)


if __name__ == '__main__':
    _main()
