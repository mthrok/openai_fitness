import os
import cPickle as pickle

import h5py
import numpy as np


MNIST = os.path.join(os.path.dirname(__file__), 'mnist.pkl')


def parse_command_line_args():
    from argparse import ArgumentParser as AP
    ap = AP(
        description=(
            'Create sample batch from MNIST dataset. '
            '(http://deeplearning.net/data/mnist/mnist.pkl.gz) '
            'Resulting batch has shape (N=10, C[=4], H[=28], W[=27])'
            'where the first index corresponds to digit.'
        )
    )
    ap.add_argument('output', help='Output file path')
    ap.add_argument('--input', help='mnist.pkl data', default=MNIST)
    ap.add_argument('--key', default='input', help='Name of dataset to create')
    ap.add_argument('--height', default=28, type=int, help='batch height.')
    ap.add_argument('--width', default=27, type=int, help='batch width.')
    ap.add_argument('--channel', default=4,
                    type=int, help='#Channels in batch.')
    ap.add_argument('--plot', action='store_true',
                    help='Visualize the resulting data')
    return ap.parse_args()


def load_data(filepath):
    return pickle.load(open(filepath, 'r'))[0]


def process(dataset, channel=4, height=28, width=27):
    b, ret, sample = 0, [], []
    for datum, label in zip(*dataset):
        if label == b:
            sample.append(datum.reshape((28, 28))[:height, :width])
        if len(sample) == channel:
            ret.append(sample)
            sample = []
            b += 1
    return np.asarray(ret, dtype=np.float32)


def plot(batch):
    import matplotlib.pyplot as plt

    n_channels = batch.shape[1]
    n_row = np.ceil(np.sqrt(n_channels))
    n_col = n_channels // n_row
    for b, sample in enumerate(batch):
        fig = plt.figure()
        for c, channel in enumerate(sample):
            ax = fig.add_subplot(n_row, n_col, c+1)
            ax.imshow(255 * channel, cmap='Greys')
            ax.set_title('Batch: {}, Channel: {}'.format(b, c))
    plt.show()


def save_data(key, data, filepath):
    f = h5py.File(filepath, 'a')
    if key in f:
        del f[key]
    f.create_dataset(key, data=data)
    f.close()


def main():
    args = parse_command_line_args()
    data = process(load_data(args.input),
                   args.channel, args.height, args.width)
    if args.plot:
        plot(data)
    save_data(args.key, data, args.output)


if __name__ == '__main__':
    main()
