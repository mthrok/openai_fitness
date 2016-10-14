import h5py


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Delete a dataset from H5 file'
    )
    ap.add_argument('input', help='Input H5 file.')
    ap.add_argument('keys', nargs='+', help='Names of dataset to delete')
    return ap.parse_args()


def main():
    args = parse_command_line_args()
    f = h5py.File(args.input, 'r+')
    for key in args.keys:
        del f[key]

if __name__ == '__main__':
    main()
