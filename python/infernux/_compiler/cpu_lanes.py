"""Small NumPy primitives used by cooked, predicated work-item programs.

The leading axis is always the active work-item axis. Remaining axes belong
to one work item (for example the three components of a vector3).
"""
from __future__ import annotations

import numpy as np


def binary(operation, left, right, left_varying, right_varying, *, output=None):
    rank = max(np.ndim(left) - int(left_varying), np.ndim(right) - int(right_varying))
    if left_varying and np.ndim(left) - 1 < rank:
        left = np.reshape(left, (len(left),) + (1,) * (rank - np.ndim(left) + 1) + np.shape(left)[1:])
    if right_varying and np.ndim(right) - 1 < rank:
        right = np.reshape(right, (len(right),) + (1,) * (rank - np.ndim(right) + 1) + np.shape(right)[1:])
    if output is not None:
        return operation(left, right, out=output, casting='unsafe')
    return operation(left, right)


def partition(count, condition):
    condition = np.asarray(condition, dtype=np.bool_)
    if condition.ndim == 0:
        whole, empty = np.arange(count, dtype=np.intp), np.empty(0, dtype=np.intp)
        return (whole, empty) if condition else (empty, whole)
    if condition.shape != (count,):
        raise TypeError("A compute branch must have one boolean per work item")
    return np.flatnonzero(condition), np.flatnonzero(np.logical_not(condition))


def selected(count, condition):
    condition = np.asarray(condition, dtype=np.bool_)
    if condition.ndim == 0:
        return np.arange(count, dtype=np.intp) if condition else np.empty(0, dtype=np.intp)
    if condition.shape != (count,):
        raise TypeError("A compute branch must have one boolean per work item")
    return np.flatnonzero(condition)


def store(previous, indices, value, varying, width, *, promote=False):
    value_shape = np.shape(value)
    tail = value_shape[1:] if varying else value_shape
    if previous is None:
        # Inactive slots have no authored value and are never read by the
        # current block; initialize their storage deterministically.
        previous = np.zeros((width, *tail), dtype=np.asarray(value).dtype)
    elif promote:
        dtype = np.result_type(previous.dtype, np.asarray(value).dtype)
        if dtype != previous.dtype:
            previous = previous.astype(dtype)
    if varying and np.ndim(value) < previous.ndim:
        value = np.reshape(value, (len(value),) + (1,) * (previous.ndim - np.ndim(value)) + value_shape[1:])
    previous[indices] = value
    return previous


def write(buffer, indices, lane, value, varying, operation=None):
    address = indices if lane is None else (indices, lane)
    if operation is not None:
        if isinstance(indices, slice) and (lane is None or isinstance(lane, (int, np.integer))):
            target = buffer[address]
            binary(operation, target, value, True, varying, output=target)
            return
        value = binary(operation, buffer[address], value, True, varying)
    elif varying:
        tail_rank = buffer.ndim - 1 - int(lane is not None)
        if np.ndim(value) - 1 < tail_rank:
            value = np.reshape(value, (len(value),) + (1,) * (tail_rank - np.ndim(value) + 1) + np.shape(value)[1:])
    buffer[address] = value


def pack(values, varying, count):
    return np.stack([value if per_item else np.broadcast_to(value, (count, *np.shape(value)))
                     for value, per_item in zip(values, varying)], axis=-1)
