#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' 
@Author: qinrj
@Description: 
@Date: 7/3/18 4:55 PM
@Contact: qinrj@lamda.nju.edu.cn or 2428921608@qq.com
'''

import numpy as np
import random
import warnings


def sample_batch_indexes(low, high, size):  # from keras-rl
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn(
            'Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class Memory(object):
    def __init__(self):
        pass

    def append(self):
        pass

    def clear(self):
        pass


class ReservoirMemory(object):  # a reservoir memory by reservoir sampling

    def __init__(self, size):
        self.size = size
        self.total = 0
        self.index = []

    def __len__(self):
        return len(self.data)

    def append(self, v, pos=None):
        if self.total < self.size:  # if not full
            self.index.append(v)
        else:  # if full, replace uniformly
            if pos is None:
                j = random.randint(0, self.total)
            else:
                j = pos
            if j < self.size:
                self.index[j] = v
        self.total += 1  # number of samples that occurred

    def sample_batch(self, batch_size):
        return sample_batch_indexes(0, len(self.data), batch_size)

    def valid_total_idx(self):  # return all the samples' indices
        return range(min(self.total, self.size))

    def total_data(self):
        return self.state, self.action

    def clear(self):
        self.state = []
        self.action = []
        self.total = 0