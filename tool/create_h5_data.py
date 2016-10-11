import h5py
import numpy as np  # noqa


def parse_command_line_args():
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            'Create simple NumPy data and save it in HDF5. \n'
            'For example, '
            '\n\n'
            '    python {} "np.zeros((3, 4, 5))" foo.h5 --key bar'
            '\n\n'
            'creates HDF5 file called "foo.h5" which contains dataset called '
            '"bar", \nof which value is zero-array with shape (3, 4, 5).'
            .format(__file__)
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument('expression', help='Expression to create data.')
    ap.add_argument('output', help='Output file name.')
    ap.add_argument('--key', help='Name of dataset in HDF5', default='data')
    return ap.parse_args()


def save(data, output_file, key='data'):
    f = h5py.File(output_file, 'a')
    if key in f:
        del f[key]
    f.create_dataset(key, data=data)
    f.close()


def main():
    args = parse_command_line_args()
    data = None
    exec('data = {}'.format(args.expression))
    save(data, args.output, args.key)


if __name__ == '__main__':
    main()
