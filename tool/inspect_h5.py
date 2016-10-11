import h5py
import numpy as np


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='List up dataset in H5 file with some statistics'
    )
    ap.add_argument('input', help='Input H5 file.')
    return ap.parse_args()


def max_str(l):
    return max(map(lambda e: len(str(e)), l))


def print_summary(summary):
    dtype_len = max_str(summary['dtype']) + 1
    shape_len = max_str(summary['shape']) + 1
    path_len = max_str(summary['path']) + 1
    print (
        '{path:{path_len}}{dtype:{dtype_len}}{shape:{shape_len}} '
        '{sum:>10}  {max:>10}  {min:>10}  {mean:>10}'
        .format(
            dtype='dtype', dtype_len=dtype_len,
            shape='shape', shape_len=shape_len,
            path='path', path_len=path_len,
            sum='sum', max='max', min='min', mean='mean'
        )
    )
    for dtype, shape, path, mean, sum, max, min in zip(
            summary['dtype'], summary['shape'], summary['path'],
            summary['mean'], summary['sum'], summary['max'], summary['min']):
        print (
            '{path:{path_len}}{dtype:{dtype_len}}{shape:{shape_len}} '
            '{sum:10.3E}  {max:10.3E}  {min:10.3E}  {mean:10.3E}'
            .format(
                dtype=dtype, dtype_len=dtype_len,
                shape=shape, shape_len=shape_len,
                path=path, path_len=path_len,
                sum=sum, max=max, min=min, mean=mean,
            )
        )


def merge_summaries(summary1, summary2):
    for key, value in summary2.items():
        summary1[key].extend(value)


def list_dataset(f, prefix=''):
    summary = {
        'dtype': [],
        'shape': [],
        'path': [],
        'mean': [],
        'sum': [],
        'max': [],
        'min': [],
    }
    for key, value in f.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(value, h5py.Group):
            merge_summaries(summary, list_dataset(value, prefix=path))
        else:
            summary['dtype'].append(value.dtype)
            summary['shape'].append(value.shape)
            summary['path'].append(path)
            summary['mean'].append(np.mean(value))
            summary['sum'].append(np.sum(value))
            summary['max'].append(np.max(value))
            summary['min'].append(np.min(value))
    return summary


def main():
    args = parse_command_line_args()
    f = h5py.File(args.input, 'r')
    print_summary(list_dataset(f))

if __name__ == '__main__':
    main()
