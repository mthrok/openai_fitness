"""Script to create simple dataset from celeba aligned images

You can use this script to generate face image dataset file
from aligned CelebA dataset.

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
"""
from __future__ import division
from __future__ import print_function

import os
import pickle

import cv2
import numpy as np


def _parse_command_line_args():
    import argparse
    default_output = 'celeba_faces.pkl'

    parser = argparse.ArgumentParser(
        description='Create dataset from CelebA face images. '
    )
    parser.add_argument(
        'directory', help='Directory which contains image files.'
    )
    parser.add_argument(
        '--width', type=int, default=64, help='Target image width.'
    )
    parser.add_argument(
        '--height', type=int, default=64, help='Target image height.'
    )
    parser.add_argument(
        '--n-train', type=int, default=50000,
        help='The number of samples in training data set.'
    )
    parser.add_argument(
        '--n-test', type=int, default=10000,
        help='The number of samples in testing data set.'
    )
    parser.add_argument(
        '--n-valid', type=int, default=10000,
        help='The number of samples in validation data set.'
    )
    parser.add_argument(
        '--seed', default=123,
        help='Seed value for randomly shaffling images.'
    )
    parser.add_argument(
        '--output', default=default_output,
        help='output file name. Default: {}'.format(default_output)
    )
    return parser.parse_args()


def _get_image_files(directory):
    return [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']
    ]


def _create_dataset(image_filenames, size):
    shape = (len(image_filenames), 3,) + size
    data = np.empty(shape, dtype=np.uint8)
    for i, filename in enumerate(image_filenames):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        res = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
        res = res[:, :, ::-1]  # BGR -> RGB
        data[i, ...] = res.transpose(2, 0, 1)  # HWC -> CHW
    print('  Created dataset;')
    print('    Shape:', data.shape)
    print('    DType:', data.dtype)
    print('    Mean:', data.mean())
    print('    Max:', data.max())
    print('    Min:', data.min())
    return data


def _create_datasets(filenames, size, n_train, n_valid, n_test):
    i_valid, i_test = n_train, n_train + n_valid
    print('Creating training dataset...')
    train_ = _create_dataset(filenames[:n_train], size)
    print('Creating validation dataset')
    valid_ = _create_dataset(filenames[i_valid:i_valid+n_valid], size)
    print('Creating test dataset...')
    test_ = _create_dataset(filenames[i_test:i_test+n_test], size)
    return ((train_, None), (valid_, None), (test_, None))


def _main():
    args = _parse_command_line_args()

    files = _get_image_files(args.directory)
    if len(files) < args.n_train + args.n_test + args.n_valid:
        raise ValueError('Not enough images were found.')

    np.random.RandomState(args.seed).shuffle(files)
    datasets = _create_datasets(
        files, (args.height, args.width),
        args.n_train, args.n_test, args.n_valid
    )
    print('Saving data to', args.output)
    with open(args.output, 'wb') as file_:
        pickle.dump(datasets, file_)


if __name__ == '__main__':
    _main()
