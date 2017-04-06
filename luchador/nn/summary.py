"""Implement SummaryWriter"""
from __future__ import absolute_import

import logging
import collections

import numpy as np
import tensorflow as tf

_LG = logging.getLogger(__name__)


__all__ = ['SummaryWriter']


SummaryOperation = collections.namedtuple('SummaryOperation', ('pf', 'op'))


def _create_summary_op(type_, name, **kwargs):
    pf = tf.placeholder('float32')
    op = getattr(tf.summary, type_)(name, pf, **kwargs)
    return SummaryOperation(pf, op)


class SummaryWriter(object):
    """Wrap tf.SummaryWriter for better flwexibility

    Unlike Tensorflow's native SummaryWrite, this SummaryWriter accepts
    NumPy Arrays for summarization. Generation of summary protocol buffer
    is handled internally.

    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.summary_ops = collections.defaultdict(dict)

        self.writer = tf.summary.FileWriter(self.output_dir)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

    ###########################################################################
    # Basic functionalitites
    def add_graph(self, graph=None, global_step=None):
        """Add graph summary. Affective only in tensorflow backend"""
        self.writer.add_graph(graph, global_step=global_step)

    def _get_summary_op(self, summary_type, name, **kwargs):
        """Fetch or create summary operation and placeholder"""
        if name in self.summary_ops[summary_type]:
            return self.summary_ops[summary_type][name]

        with self.graph.as_default():
            with self.graph.device('/cpu:0'):
                op = _create_summary_op(summary_type, name, **kwargs)
                self.summary_ops[summary_type][name] = op
        return op

    def summarize(self, summary_type, global_step, dataset, **kwargs):
        """Summarize the dataset

        Parameters
        ----------
        summary_type : str
            Type of summary to create. ``scalar``, ``histogram``, ``image``
            and ``audio`` are supported.

        global_step : int
            Global step value to record with the summary.

        dataset : dict
            key : str
                Summary name
            value : NumPy NDArray
                Value to summarize. The shape of each array must be in
                accordance with summary type.

        **kwargs
            Other keyward argument fed to summary operation function.
            For ``image`` summary, ``max_outputs`` is accepted.
            For ``audio`` summary, ``sample_rate`` must be given and
            ``max_outputs`` is optional.

            See tf.summary.image or tf.summary.audio for the detail.
        """
        ops, feed_dict = [], {}
        for name, value in dataset.items():
            summary_op = self._get_summary_op(summary_type, name, **kwargs)
            ops.append(summary_op.op)
            feed_dict[summary_op.pf] = value

        summaries = self.session.run(ops, feed_dict=feed_dict)
        for summary in summaries:
            self.writer.add_summary(summary, global_step)
        self.writer.flush()

    ###########################################################################
    def summarize_stats(self, global_step, dataset):
        """Summarize max/min/average of the given dataset

        Parameters
        ----------
        global_step : int
            Global step used as surffix for output file name

        dataset : dict
            Key : str
                Names used in :any:`register_stats`
            Value : list of floats, or NumPy Array
                Values to summarize stats
        """
        _dataset = {}
        for name, values in dataset.items():
            _dataset['{}/Average'.format(name)] = np.mean(values)
            _dataset['{}/Min'.format(name)] = np.min(values)
            _dataset['{}/Max'.format(name)] = np.max(values)
        self.summarize(
            summary_type='scalar', global_step=global_step, dataset=_dataset)
