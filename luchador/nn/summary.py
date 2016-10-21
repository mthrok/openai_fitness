from __future__ import absolute_import

import logging

import numpy as np
import tensorflow as tf

_LG = logging.getLogger(__name__)


__all__ = ['SummaryWriter']


class SummaryOperation(object):
    """Create placeholder and summary operations for the given names"""
    def __init__(self, type, name):
        summary_funcs = {
            'scalar': tf.scalar_summary,
            'image': tf.image_summary,
            'audio': tf.audio_summary,
            'histogram': tf.histogram_summary,
        }
        self.pf = tf.placeholder('float32')
        self.op = summary_funcs[type](name, self.pf)


class SummaryWriter(object):
    def __init__(self, output_dir, graph=None):
        self.output_dir = output_dir
        self.summary_ops = {}

        self.writer = tf.train.SummaryWriter(self.output_dir)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    ###########################################################################
    # Basic functionalitites
    def add_graph(self, graph=None, global_step=None):
        self.writer.add_graph(graph, global_step=global_step)

    def register(self, type, names, tag=None):
        with self.graph.as_default():
            with self.graph.device('/cpu:0'):
                self._register(type, names)

    def _register(self, type, names, tag=None):
        for name in names:
            self.summary_ops[name] = SummaryOperation(type, name)

    def summarize(self, global_step, dataset):
        """Summarize the dataset

        Args:
          global_step (int): Global step

          dataset (dict):
            Keys correspond to registered summary operation and
            values correspond to the actual values to summaryze
        """
        ops, feed_dict = [], {}
        for name, value in dataset.items():
            ops.append(self.summary_ops[name].op)
            feed_dict[self.summary_ops[name].pf] = value

        summaries = self.session.run(ops, feed_dict=feed_dict)
        for summary in summaries:
            self.writer.add_summary(summary, global_step)
        self.writer.flush()

    ###########################################################################
    # Convenient functions
    def register_stats(self, names):
        """For each name, create 'name/[Average, Min, Max]' summary ops"""
        all_names = ['{}/{}'.format(name, stats) for name in names
                     for stats in ['Average', 'Min', 'Max']]
        self.register('scalar', all_names)

    def summarize_stats(self, global_step, dataset):
        """Summarize statistics of dataset

        Args:
          global_step (int): Global step

          dataset (dict):
            - Key (str): Names used in `register_stats`
            - Value (list of floats, or NumPy Array): Values to summarize stats
        """
        for name, values in dataset.items():
            _dataset = {
                '{}/Average'.format(name): np.mean(values),
                '{}/Min'.format(name): np.min(values),
                '{}/Max'.format(name): np.max(values)
            }
            self.summarize(global_step, _dataset)
    ###########################################################################
