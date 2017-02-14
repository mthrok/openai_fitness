"""Implements prioritized replay memory [ARXIV05952]_

References
----------
.. [ARXIV05952] Tom Schaul, John Quan, Ioannis Antonoglou, David Silver (2015):
       Prioritized Experience Replay
       https://arxiv.org/abs/1511.05952
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np


def _compute_partition_index(buffer_size, sample_size, priority):
    """Compute partition indices for rank-based prioritization

    Given N memory slots in array format, rank :math:`r(i)` and priority
    :math:`p(i)`of the element at index i (0-based index) is

    .. math::
        r(i) &= i + 1
        p(i) &= r(i)^{-\\alpha} = (i + 1) ^ {-\\alpha}

    where :math:`\\alpha` is prioritization weight.

    This function returns the indices which partitions cumulative priorities
    into equal value bins

    Parameters
    ----------
    buffer_size : int
        The size of memory slots

    sample_size : int
        #partitions to separate priorities

    priority : float
        Priority weight parameter, :math:`\\alpha`. Must be between [0, 1.0]
        1 is full prioritized, 0 is no prioritization is effective, sampling is
        uniformly random.

    Returns
    -------
    list of tuple of two ints
        Each tuple is starting index and ending index of partition

    NumPy ND Array
        N-length array contains sapmling probability.
    """
    rank = np.arange(1, buffer_size+1)
    priority = np.power(rank, -priority)
    p_cumsum = np.cumsum(priority)
    probabilities = priority / p_cumsum[-1]
    p_cumsum -= p_cumsum[0]
    p_unit = p_cumsum[-1] / (sample_size + 1)

    indices, i_start = [], 0
    for _ in range(sample_size):
        p_cumsum -= p_unit
        i_end = max(i_start + 1, np.where(np.diff(np.sign(p_cumsum)))[0][0])
        indices.append((i_start, i_end))
        i_start = i_end
    return indices, probabilities


def _get_child_index(index, buffer_):
    """Get a valid child index of the given index in binary tree.

    Returns
    -------
    int or None
        If the range of child indices are out of buffer length, None is
        returned, else the index of the child with larger priority is
        returned.
    """
    i1, i2 = [2 * index + i + 1 for i in range(2)]
    length = len(buffer_)
    if i2 >= length:
        if i1 >= length:
            return None
        return i1
    return i1 if buffer_[i1] > buffer_[i2] else i2


def _get_parent_index(index):
    return (index - 1) // 2


class _PriorityRecord(object):  # pylint: disable=too-few-public-methods
    """Record with priority comparison"""
    def __init__(self, priority, record_id):
        self.priority = priority
        self.record_id = record_id

    def __repr__(self):
        return '(Priority: {}, Record: {})'.format(
            self.priority, self.record_id)

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __ge__(self, other):
        return self.priority >= other.priority


class PrioritizedQueue(object):
    # pylint: disable=line-too-long, too-many-instance-attributes
    """Fixed-size max priority queue

    Parameters
    ----------
    buffer_size : int
        The size of maximum records

    sample_size : int
        Size of sample mini-batch

    priority : float
        Priority parameter, :math:`\\alpha` in [ARXIV05952]_. Must be in the
        range of [0.0, 1.0]. When 1, sampling is fully weighted by priorities.
        When 0, priority is notin effect thus sampling is just uniformly
        random.

    importance : float
        Importance sampling weight. :math:`\\beta` in [ARXIV05952]_

    random_seed
        Random seed value for sampling records
    """
    def __init__(
            self, buffer_size, sample_size, priority,
            importance, random_seed=None):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.importance = importance
        self.priority = priority
        self._rng = np.random.RandomState(seed=random_seed)

        # The actual heap buffer
        self.buffer = []
        indices, probabilities = _compute_partition_index(
            self.buffer_size, self.sample_size, self.priority)
        self.partition_indices = indices
        self.probabilities = probabilities

        # Record IDs are used for tracking the oldest records and locate them
        # in heap buffer.
        # Since the incoming record is not guaranteed to be hashable, we use
        # record ID and store the actual record separately.
        self.new_record_id = 0
        self.update_record_id = 0

        self.id2record = {}  # record ID -> actual record
        self.id2index = {}  # record ID -> index in array

    ###########################################################################
    # Hepler methods for balancing/sorting heap
    def _swap(self, i_1, i_2):
        """Swap records and update index mapping"""
        self.buffer[i_1], self.buffer[i_2] = self.buffer[i_2], self.buffer[i_1]
        self.id2index[self.buffer[i_2].record_id] = i_2
        self.id2index[self.buffer[i_1].record_id] = i_1

    def _balance_up(self, index):
        """Recursively move up a record if its priority is higher

        Parameters
        ----------
        index : int
            Index of record to move up

        Returns
        -------
        bool
            True if swap is performed.
        """
        if index == 0:
            return False

        i_parent = _get_parent_index(index)
        if self.buffer[i_parent] >= self.buffer[index]:
            return False

        self._swap(i_parent, index)
        self._balance_up(i_parent)
        return True

    def _balance_down(self, index):
        """Recursively move down a record if its priority is lower

        Parameters
        ----------
        index : int
            Index of record to move up

        Returns
        -------
        bool
            True if swap is performed.
        """
        i_child = _get_child_index(index, self.buffer)
        if i_child is None:
            return False

        if self.buffer[index] >= self.buffer[i_child]:
            return False

        self._swap(index, i_child)
        self._balance_down(i_child)
        return True

    ###########################################################################
    # Method for pushing new record
    def push(self, priority, record):
        """Push new record to heap.

        Parameters
        ----------
        priority : float
            Priority of new record. Must be positive.

        record
            Record to push

        Returns
        -------
        int
            Index of the resulting record location

        Examples
        --------
        >>> buffer_size = 3
        >>> queue = PrioritizedQueue(buffer_size=buffer_size, ...)
        >>> for i in range(buffer_size):
        >>>     queue.push(priority=i, record=None)
        >>>     print(queue.buffer)
        [(Priority: 0, ...)]
        [(Priority: 1, ...), (Priority: 0, ...)]
        [(Priority: 2, ...), (Priority: 0, ...), (Priority: 1, ...)]
        # Now the #stored records reached its limit, old records are removed
        # automatically in the following `push`
        >>> for i in range(buffer_size, 2 * buffer_size):
        >>>     queue.push(priority=i, record=None)
        >>>     print(queue.buffer)
        [(Priority: 3, ...), (Priority: 2, ...), (Priority: 1, ...)]
        [(Priority: 4, ...), (Priority: 2, ...), (Priority: 3, ...)]
        [(Priority: 5, ...), (Priority: 4, ...), (Priority: 3, ...)]
        """
        index = len(self.buffer)
        if index >= self.buffer_size:
            self._update_push(priority, record)
        else:
            self._append_push(priority, record)

    def _append_push(self, priority, record):
        """Push a record by appending it at the end of buffer"""
        new_id = self.new_record_id
        self.new_record_id += 1

        # Add new <ID -> record> mapping
        self.id2record[new_id] = record

        # Add new priority record to heap buffer
        self.buffer.append(_PriorityRecord(priority, new_id))

        # Add new <ID -> index> mapping
        index = len(self.buffer) - 1
        self.id2index[new_id] = index

        # Move up if necessary
        self._balance_up(index)

    def _update_push(self, priority, record):
        """Push a record by overwriting the oldest record"""
        old_id, new_id = self.update_record_id, self.new_record_id
        self.update_record_id += 1
        self.new_record_id += 1

        # Add new <ID -> record> mapping and remove old one
        self.id2record[new_id] = record
        del self.id2record[old_id]

        # Update the content of old record in heap buffer with new one
        index = self.id2index[old_id]
        self.buffer[index] = _PriorityRecord(priority, new_id)

        # Add new <ID -> index> mapping and remove old one
        self.id2index[new_id] = index
        del self.id2index[old_id]

        # Move up or down if necessary
        if self._balance_up(index):
            return
        self._balance_down(index)

    ###########################################################################
    # Record retrievals
    def sample(self):
        """Sample records from buffer

        Returns
        -------
        dict
            records : list
                Records
            weights : list
                Sampling weights of each records
            indices : list
                Indices where records were stored in buffer. To be used to
                update priorities later.
        """
        buffer_size = len(self.buffer)
        if buffer_size < self.buffer_size:
            partitions, probabilities = _compute_partition_index(
                buffer_size, self.sample_size, self.priority)
        else:
            partitions = self.partition_indices
            probabilities = self.probabilities

        indices = [self._rng.randint(i0, i1) for i0, i1 in partitions]
        record_ids = [self.buffer[i].record_id for i in indices]
        records = [self.id2record[id_] for id_ in record_ids]
        probs = probabilities[indices]
        weights = np.power(1 / probs / buffer_size, self.importance)
        weights /= np.max(weights)
        weights = weights.astype('float32')
        return {'indices': indices, 'records': records, 'weights': weights}

    def get_last_record(self):
        """Get the record inserted previously

        Returns
        -------
        record
            Record added with previous push
        """
        return self.id2record[self.new_record_id - 1]

    ###########################################################################
    def update(self, indices, priorities):
        """Update priority of records and balance tree

        Parameters
        ----------
        indices : list
            List of indices to update priority

        priorities : list
            New priority values
        """
        for index, priority in zip(indices, priorities):
            self.buffer[index].priority = priority
            if not self._balance_up(index):
                self._balance_down(index)

    ###########################################################################
    # Quick sort
    def sort(self):
        """Sort internal buffer"""
        self._quick_sort(0, len(self.buffer) - 1)

    def _partition(self, i_start, i_end):
        pivot = (
            self.buffer[i_start].priority + self.buffer[i_end].priority
        ) / 2

        i_left, i_right = i_start, i_end
        while i_left < i_right:
            while self.buffer[i_left].priority > pivot:
                i_left += 1
            while self.buffer[i_right].priority < pivot:
                i_right -= 1
            if i_left < i_right:
                self._swap(i_left, i_right)
            if i_left <= i_right:
                i_left += 1
                i_right -= 1
        return i_right, i_left

    def _quick_sort(self, i_start, i_end):
        if i_start < i_end:
            i_end_, i_start_ = self._partition(i_start, i_end)
            self._quick_sort(i_start, i_end_)
            self._quick_sort(i_start_, i_end)
