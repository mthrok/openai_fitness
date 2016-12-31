"""Implement SummaryWriter"""

from __future__ import absolute_import

import logging
import collections

import numpy as np
import tensorflow as tf

_LG = logging.getLogger(__name__)


__all__ = ['SummaryWriter']


SummaryOperation = collections.namedtuple('SummaryOperation', ('pf', 'op'))


def _create_summary_op(type_, name):
    summary_funcs = {
        'scalar': tf.summary.scalar,
        'image': tf.summary.image,
        'audio': tf.summary.audio,
        'histogram': tf.summary.histogram,
    }
    pf = tf.placeholder('float32')
    op = summary_funcs[type_](name, pf)
    return SummaryOperation(pf, op)


class SummaryWriter(object):
    """Wrap tf.SummaryWriter for better flwexibility

    Unlike Tensorflow's native SummaryWrite, this SummaryWriter accepts
    NumPy Arrays for summarization. Generation of summary protocol buffer
    is handled internally.

    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.summary_ops = {}
        self.tags = {}

        self.writer = tf.summary.FileWriter(self.output_dir)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    ###########################################################################
    # Basic functionalitites
    def add_graph(self, graph=None, global_step=None):
        self.writer.add_graph(graph, global_step=global_step)

    def register(self, summary_type, names, tag=None):
        with self.graph.as_default():
            with self.graph.device('/cpu:0'):
                self._register(summary_type, names, tag=tag)

    def _register(self, summary_type, names, tag):
        for name in names:
            self.summary_ops[name] = _create_summary_op(summary_type, name)

        if tag:
            self.tags[tag] = names

    def summarize(self, global_step, dataset, tag=None):
        """Summarize the dataset

        Parameters
        ----------
        global_step : int
            Global step used as surffix for output file name

        dataset : (dict or list of values)
            When tag is not given this must be a dictionary mapping name of
            registered summary operation to summary value.
            When tag is given, summary operations registered with the tag are
            pulled, so only values are needed so dataset must be a list of
            values in the same order as the operations registered.

        tag : str
            See above
        """
        ops, feed_dict = [], {}
        if tag:
            dataset = {name: val for name, val in zip(self.tags[tag], dataset)}
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
        """For each name, create ``name/[Average, Min, Max]`` summary ops"""
        all_names = ['{}/{}'.format(name, stats) for name in names
                     for stats in ['Average', 'Min', 'Max']]
        self.register('scalar', all_names, tag=None)

    def summarize_stats(self, global_step, dataset):
        """Summarize statistics of dataset

        Parameters
        ----------
        global_step : int
            Global step used as surffix for output file name

        dataset (dict):
            Key : str
                Names used in :any:`register_stats`
            Value : list of floats, or NumPy Array
                Values to summarize stats
        """
        for name, values in dataset.items():
            _dataset = {
                '{}/Average'.format(name): np.mean(values),
                '{}/Min'.format(name): np.min(values),
                '{}/Max'.format(name): np.max(values)
            }
            self.summarize(global_step, _dataset, tag=None)
    ###########################################################################
