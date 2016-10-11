from __future__ import print_function

import h5py
import numpy as np

import matplotlib.pyplot as plt


def parse_command_line_args():
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            'Visualize the datasets in NCHW format (batch, channel, height, \n'
            'width) from different HDF5 files side by side. \n'
            '\n'
            'This script was written to debug Layer IO compatibility test \n'
            'and a typical usage is as follow. \n'
            '\n'
            '    python {} tmp/'
            'test_layer_numerical_comparitbility_conv2d_valid_tensorflow.h5 '
            'tmp/test_layer_numerical_comparitbility_conv2d_valid_theano.h5'
            '\n\n'.format(__file__)
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument('files', nargs='+', help='Output files from test.')
    ap.add_argument(
        '--key', default='output',
        help='The name of the dataset in the given H5 file. Default: "output"'
    )
    return ap.parse_args()


def load(files, key):
    ret = []
    for f in files:
        print('Loading "{}" from {}'.format(key, f))
        f = h5py.File(f, 'r')
        data = np.asarray(f[key])
        f.close()
        print('  Shape: {}'.format(data.shape))
        print('    Ave: {}'.format(data.mean()))
        print('    Max: {}'.format(data.max()))
        print('    Min: {}'.format(data.min()))
        ret.append(data)
    return ret


def visualize(filenames, data):
    v_min = min(map(np.min, data))
    v_max = max(map(np.max, data))

    n_data = len(data)
    n_batches, n_channels = data[0].shape[:2]
    for b in range(n_batches):
        fig = plt.figure()
        i = 1
        for c in range(n_channels):
            for datum, f in zip(data, filenames):
                sample = datum[b]
                axis = fig.add_subplot(n_channels, n_data, i)
                axis.imshow(sample[c], cmap='Greys', vmin=v_min, vmax=v_max)
                axis.set_title('{}\nC:{}, B:{}'.format(f, c, b))
                i += 1
    plt.show()


def main():
    args = parse_command_line_args()
    data = load(args.files, args.key)
    visualize(args.files, data)

if __name__ == '__main__':
    main()
