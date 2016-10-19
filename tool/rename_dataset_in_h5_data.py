import h5py


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description='Rename a dataset in H5 file'
    )
    ap.add_argument('input', help='Input H5 file.')
    ap.add_argument('old_name', help='Dataset to rename')
    ap.add_argument('new_name', help='New Dataset name')
    return ap.parse_args()


def main():
    args = parse_command_line_args()
    f = h5py.File(args.input, 'r+')
    f[args.new_name] = f[args.old_name]
    del f[args.old_name]

if __name__ == '__main__':
    main()
