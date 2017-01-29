"""Module for implelemting customizable ReplayMemory mechanism"""
from __future__ import division
from __future__ import absolute_import

import logging
from collections import deque

import numpy as np

_LG = logging.getLogger(__name__)


class Buffer(object):
    """Handles recording and sampling of data

    Parameters
    ----------
    stack : int
        The number of datum included in one stack

    stansition : int
        The number of stacked data to fetch
    """
    def __init__(self, stack, transition):
        self.data = []
        self.stack = stack
        self.transition = transition

    def put(self, datum):
        """Append data to the end"""
        self.data.append(datum)

    def _get(self, index):
        """Fetch data from buffer and stack

        Parameters
        ----------
        index : int
            Index for the last datum to include in the resulting data.

            For performance reason, argument validity check is omitted, but
            arguments must satisfy the following condition:

                0 <= stack - 1 <= index < len(self.data)

        Returns
        -------
        list
            data stored in the corresponding index
        """
        if self.stack:
            i_start, i_end = index - self.stack + 1, index + 1
            return self.data[i_start:i_end]
        else:
            return self.data[index]

    def get(self, index):
        """Fetch multiple stacked data from buffer

        Parameters
        ----------
        index : int
            Index for the last datum to include in the first data to be
            fetched.

            This index has stronger restriction than ``_get`` method,
            as this method fetches multiple stacked data.

                0 <= stack - 1 <= index < len(self.data) - self.transition

        """
        if self.transition:
            return [self._get(index+i) for i in range(self.transition)]
        else:
            return self._get(index)

    def get_last_stack(self):
        """Fetch the latest data and stack if necessary"""
        if self.stack:
            return self.data[-self.stack:]
        else:
            return self.data[-1]

    def __str__(self):
        return 'Buffer: <stack: {}, transition: {} > (#Records: {})'.format(
            self.stack, self.transition, len(self.data))


class EpisodeRecorder(object):
    """Record/Sample multiple data stream for an episode

    Parameters
    ----------
    buffer_configs : dict
        key is the name of data to record, value is dict containing
        'stack' and 'transition', which are passed to Buffer constructor

    initial_data : dict
        Initial data to stored in recorders. key is the name of recorder
    """
    def __init__(self, buffer_configs, initial_data):
        self.buffers = {
            name: Buffer(
                stack=cfg.get('stack'), transition=cfg.get('transition'))
            for name, cfg in buffer_configs.items()
        }
        self.n_data = 0

        for name, data in initial_data.items():
            self.buffers[name].put(data)

        # Max stack size and transition size of underlying buffer.
        # These are used to compute valid sampling index
        self.max_stack = max([
            buf.stack for buf in self.buffers.values()])
        self.max_transition = max([
            buf.transition for buf in self.buffers.values()])

    def put(self, data):
        """Put data to underlying buffers"""
        for name, buffer_ in self.buffers.items():
            buffer_.put(data[name])
        self.n_data += 1

    def get(self, index):
        """Get data from underlying buffers"""
        return {
            name: buffer_.get(index)
            for name, buffer_ in self.buffers.items()
        }

    def enough_data(self):
        """Check if there is enough data to create transition"""
        return self.n_data > self.max_stack + self.max_transition

    def sample(self):
        """Sample transition randomly"""
        index = np.random.randint(
            self.max_stack - 1, self.n_data - self.max_transition + 1)
        return self.get(index)

    def get_last_stack(self):
        """Get the latest record stack"""
        return {
            name: buffer_.get_last_stack()
            for name, buffer_ in self.buffers.items()
        }

    def __str__(self):
        return 'EpisodeRecorder:\n' + '\n'.join([
            '  {}: {}'.format(name, buffer_)
            for name, buffer_ in self.buffers.items()
        ])


class TransitionRecorder(object):
    """Record/Sample records across episodes

    Parameters
    ----------
    memory_size : int
        The number of records to retain.

    buffer_config : dict
        See :class:`EpisodeRecorder`

    batch_size : int
        Buffer size for sampling
    """
    def __init__(self, memory_size, buffer_config, batch_size=32):
        self.memory_size = memory_size
        self.buffer_config = buffer_config
        self.batch_size = batch_size

        self._recorder = None
        self._recorders = deque()

        self._batch = {}
        self._init_batch()

    def _init_batch(self):
        for name, cfg in self.buffer_config.items():
            shape = list(cfg.get('shape', []))
            stack = cfg.get('stack')
            transition = cfg.get('transition')
            if stack:
                shape = [stack] + shape
            shape = [self.batch_size] + shape
            if transition:
                self._batch[name] = [
                    np.empty(shape=shape, dtype=cfg['dtype'])
                    for _ in range(cfg['transition'])
                ]
            else:
                self._batch[name] = np.empty(
                    shape=shape, dtype=cfg['dtype'])

    def reset(self, initial_data):
        """Renew recorder for the current episode"""
        self._recorder = EpisodeRecorder(self.buffer_config, initial_data)

    def record(self, data):
        """Record data

        Parameters
        ----------
        data : dict
            keys are the name of buffers and
            values are the actual data to put in buffer
        """
        self._recorder.put(data)

    @property
    def n_records(self):
        """Get the number of records in archive. Not include current episode"""
        return sum([recorder.n_data for recorder in self._recorders])

    def truncate(self):
        """Put current recorder in archive and discard old recordes
        which surpasses the defined memory size"""
        self._recorders.append(self._recorder)
        self._recorder = None

        n_records = self.n_records
        n_recorders = len(self._recorders)

        _LG.debug('  Before truncate:')
        _LG.debug('  # records  : %s', n_records)
        _LG.debug('  # recorders: %s', n_recorders)

        while n_recorders > 1 and n_records > self.memory_size:
            poped = self._recorders.popleft()
            n_records -= poped.n_data
            n_recorders -= 1

        _LG.debug('  After truncate:')
        _LG.debug('  # records  : %s', n_records)
        _LG.debug('  # recorders: %s', n_recorders)

    def _set_batch(self, data, i_batch):
        for name in self._batch:
            transition = self.buffer_config[name].get('transition')
            if transition:
                for j in range(transition):
                    self._batch[name][j][i_batch] = data[name][j]
            else:
                self._batch[name][i_batch] = data[name]

    def sample(self, n_samples):
        """Sample transitions from the past episodes

        Parameters
        ----------
        n_samples : int
            Must be smaller than or equal to batch size

        Returns
        -------
        dict
            Contains sampled data
        """
        n_recorders = len(self._recorders)
        i_batch = 0
        while i_batch < n_samples:
            recorder = self._recorders[np.random.randint(n_recorders)]
            if recorder.enough_data():
                data = recorder.sample()
                self._set_batch(data, i_batch)
                i_batch += 1
        return self._batch

    def get_last_stack(self):
        """Get the latest state"""
        data = self._recorder.get_last_stack()
        ret = {}
        for name in self._batch:
            transition = self.buffer_config[name].get('transition')
            if transition:
                bucket = self._batch[name][0]
            else:
                bucket = self._batch[name]
            bucket[0] = data[name]
            ret[name] = bucket[0]
        return ret

    def is_ready(self):
        """True if the current episode recorder has sufficient records"""
        return self._recorder.enough_data()

    def __str__(self):
        return (
            'TransitionRecorder:\n'
            'Archived recorders:\n'
            '  #recorders: {}\n'
            '  #records: {})\n'
            'Current recorder:\n'
            '{}'
        ).format(
            len(self._recorders), self.n_records, self._recorder
        )
