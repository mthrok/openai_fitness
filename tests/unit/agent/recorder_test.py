"""Test prioritized replay memory"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import unittest

import numpy as np

from luchador.agent.recorder import PrioritizedQueue


def _get_rng(seed=123):
    return np.random.RandomState(seed=seed)


class TestMaxHeap(unittest.TestCase):
    """Test if PrioritizedQueue works correctly"""
    def _assert_balance(self, heap):
        """Verify that elements have larger priority than their children"""
        length = len(heap.buffer)
        for i_parent in range(length):
            for i in range(2):
                i_child = i_parent * 2 + i + 1
                if i_child >= length:
                    break
                self.assertGreaterEqual(
                    heap.buffer[i_parent], heap.buffer[i_child],
                    'Priority is unbalanced at {}({}) and {}({})'
                    .format(
                        i_parent, heap.buffer[i_parent],
                        i_child, heap.buffer[i_child],
                    )
                )

    def _assert_index_mapping(self, heap):
        """Verify id2index mapping"""
        for index, record in enumerate(heap.buffer):
            id_ = record.record_id
            index_ = heap.id2index[id_]
            self.assertEqual(
                index, index_,
                'Index mapping is not correct at {}'
                .format(index)
            )

    def _verify_heap(self, heap):
        self._assert_balance(heap)
        self._assert_index_mapping(heap)

    def test_insert(self):
        """Push inserts/updates record while keeping balance"""
        buffer_size = 1000
        sample_size = 32

        heap = PrioritizedQueue(
            buffer_size=buffer_size, sample_size=sample_size,
            priority=0.7, importance=0.5)
        for i in range(buffer_size):
            heap.push(priority=i, record=i)
            # Check data is inserted correctly
            self.assertEqual(heap.buffer[0].priority, i)
            # Check buffer is expanding
            self.assertEqual(len(heap.buffer), i + 1)
            self.assertEqual(len(heap.id2index), i + 1)
            # Check max heap property
            self._verify_heap(heap)

        for i in range(buffer_size, 2 * buffer_size):
            heap.push(priority=i, record=i)
            # Check data is inserted correctly
            self.assertEqual(heap.buffer[0].priority, i)
            # Check buffer is not expanding
            self.assertEqual(len(heap.buffer), buffer_size)
            self.assertEqual(len(heap.id2index), buffer_size)
            # Check only the last #`buffer_size` records are in memory
            for j in range(buffer_size):
                self.assertIn(i - j, heap.id2index)
                self.assertIn(i - j, heap.id2record)
                self.assertEqual(i - j, heap.id2record[i - j])
            # Check max heap property
            self._verify_heap(heap)

    def test_sort(self):
        """Sort records in priority order"""
        buffer_size, rng = 1000, _get_rng(None)

        heap = PrioritizedQueue(
            buffer_size=buffer_size, sample_size=32,
            priority=0.7, importance=0.5)

        record = 0
        for _ in range(10):
            for _ in range(buffer_size):
                heap.push(priority=rng.rand(), record=record)
                record += 1
            heap.sort()
            self._verify_heap(heap)
            for i in range(len(heap.buffer) - 1):
                self.assertGreater(heap.buffer[i], heap.buffer[i+1])

    def test_sort_sorted(self):
        """Sort does not hang when sorted"""
        buffer_size = 1000

        heap = PrioritizedQueue(
            buffer_size=buffer_size, sample_size=32,
            priority=0.7, importance=0.5)

        record = 0
        for _ in range(10):
            for _ in range(buffer_size):
                heap.push(priority=1, record=record)
                record += 1
            heap.sort()
            self._verify_heap(heap)
            for i in range(len(heap.buffer) - 1):
                self.assertGreaterEqual(heap.buffer[i], heap.buffer[i+1])

    def test_update(self):
        """Update record priority keeps the heap balance"""
        buffer_size = 30

        heap = PrioritizedQueue(
            buffer_size=buffer_size, sample_size=32,
            priority=0.7, importance=0.5)

        priority = 0
        for _ in range(buffer_size):
            heap.push(priority=priority, record=priority)
            priority += 1

        # Update the last element to have the largest priority.
        # It should be moved to the first element
        heap.update([buffer_size-1], [2 * buffer_size])
        self.assertEqual(heap.buffer[0].priority, 2 * buffer_size)
        self._verify_heap(heap)
        # Update the first element to have the smallest priority.
        # It should be moved to the first element
        heap.update([0], [-1])
        self.assertEqual(heap.buffer[-1].priority, -1)
        self._verify_heap(heap)

    def test_sampling_time(self):
        """Sampling is fast enough"""
        buffer_size, rng = int(1e6), _get_rng()

        heap = PrioritizedQueue(
            buffer_size=buffer_size, sample_size=32,
            priority=0.7, importance=0.5)

        for i in range(buffer_size):
            heap.push(priority=rng.rand(), record=i)
        t0 = time.time()
        for i in range(buffer_size, 2 * buffer_size):
            heap.push(priority=rng.rand(), record=i)
        dt1 = (time.time() - t0) / buffer_size

        n_repeats = 1000
        t0 = time.time()
        for _ in range(n_repeats):
            heap.sample()
        dt2 = (time.time() - t0) / n_repeats
        print('Push: {} [sec], Sample: {} [sec]'.format(dt1, dt2))

        threshold = 0.0005
        self.assertTrue(
            dt2 < threshold,
            'Sampling is taking too much time ({} sec). '
            'Review implementation.'.format(dt2)
        )

    def test_sorting_time(self):
        """Sorting is fast enough"""
        buffer_size, rng = int(1e6), _get_rng()

        heap = PrioritizedQueue(
            buffer_size=buffer_size, sample_size=32,
            priority=0.7, importance=0.5)

        n_repeats = 3
        dt = 0
        for _ in range(n_repeats):
            for i in range(buffer_size):
                heap.push(priority=rng.rand(), record=i)

            t0 = time.time()
            heap.sort()
            dt += (time.time() - t0) / n_repeats
        print('Sort: {} [sec]'.format(dt))

        threshold = 30
        self.assertTrue(
            dt < threshold,
            'Sort is taking too much time ({} sec). '
            'Review implementation.'.format(dt)
        )
