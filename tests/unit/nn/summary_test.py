from __future__ import absolute_import

import os
import numpy as np

from luchador.nn import SummaryWriter
from tests.unit.fixture import TestCase

OUTPUT_DIR = os.path.join('tmp', 'summary_writer_test')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# pylint: disable=invalid-name


def _list_files(prefix, dir_name=OUTPUT_DIR):
    if not os.path.exists(dir_name):
        return []
    return [
        os.path.join(dir_name, f)
        for f in os.listdir(dir_name) if f.startswith(prefix)
    ]


def _remove_files(dir_name=OUTPUT_DIR):
    for file_ in _list_files('', dir_name):
        os.remove(file_)


def _gen_random_value():
    return np.random.randint(low=0, high=2, size=(32, 8, 8, 4))


class SummaryWriterTest(TestCase):
    """Test case for SummaryWriter.

    Since it is difficult to check the event file produced by Tensorflow native
    summary operations, this test only checks if they run without crush.
    """
    def _get_empty_dir(self):
        output_dir = os.path.join(OUTPUT_DIR, self.id().split('.')[-1])
        _remove_files(output_dir)
        return output_dir

    def test_scalar_summarize(self):
        """`summarize` function summarize scalar"""
        output_dir = self._get_empty_dir()
        writer = SummaryWriter(output_dir)

        key = 'test_scalar'
        for i in range(10):
            value = i + np.random.rand()
            writer.summarize(
                summary_type='scalar', global_step=i, dataset={key: value})

    def test_histogram_summarize(self):
        """`summarize` function summarize histogram"""
        output_dir = self._get_empty_dir()
        writer = SummaryWriter(output_dir)

        for i in range(10):
            dataset = {'test_histogram': i + i * np.random.rand(32, 8, 8, 4)}
            writer.summarize(
                summary_type='histogram', global_step=i, dataset=dataset)

    def test_image_summary(self):
        """`summarize` function summarize image"""
        output_dir = self._get_empty_dir()
        writer = SummaryWriter(output_dir)

        for i in range(10):
            dataset = {'test_image': i + i * np.random.rand(32, 8, 8, 4)}
            writer.summarize(
                summary_type='image', global_step=i, dataset=dataset)

    def test_audio_summary(self):
        """`summarize` function summarize audio"""
        output_dir = self._get_empty_dir()
        writer = SummaryWriter(output_dir)

        for i in range(10):
            dataset = {'test_audio': np.random.randn(32, 1440)}
            writer.summarize(
                summary_type='audio',
                global_step=i, dataset=dataset, sample_rate=144)

    def test_stats_summary(self):
        """`summarize_stats` function summarize max/min/agerage"""
        output_dir = self._get_empty_dir()
        writer = SummaryWriter(output_dir)

        for i in range(10):
            dataset = {
                'test_stats_1': np.random.randn(100) - i,
                'test_stats_2': np.random.randn(100),
                'test_stats_3': np.random.randn(100) + i,
            }
            writer.summarize_stats(global_step=i, dataset=dataset)
