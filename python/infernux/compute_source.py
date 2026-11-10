"""Deferred typed buffer values, independent of any graph or execution queue."""
import math

import numpy as np

from infernux.execution import capture_execution_scope


class BufferSource:
    """One producer's deferred value with immutable storage once ready.

    Requests retain the producer lease, separately from caller cancellation.
    """
    def __init__(self, lease):
        self._lease = lease
        self._shape, self._dtype, self._device = tuple(lease.shape), np.dtype(lease.dtype), lease.device
        self._scope = capture_execution_scope()
        if self._scope is not None:
            self._scope._resources.add(self)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return False

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def element_count(self):
        return math.prod(self.shape)

    def _detach(self):
        if self._scope is not None:
            self._scope._resources.discard(self)
            self._scope = None

    def _pin(self):
        if self._lease is None:
            raise ReferenceError('buffer source is closed')
        return self._lease

    @property
    def done(self):
        done = self._lease is None or self._lease.done
        if done:
            self._detach()
        return done

    @property
    def ready(self):
        return self.done and self._lease is not None and self._lease.ready

    @property
    def error(self):
        return 'buffer source is closed' if self._lease is None else self._lease.error

    @property
    def identity(self):
        return self._pin().identity

    def buffer(self):
        if not self.ready:
            raise RuntimeError(self.error or 'buffer producer has not published a value')
        return self._lease.acquire()

    def close(self):
        self._lease = None
        self._detach()

    def cancel(self):
        cancel = getattr(self._lease, 'cancel', None)
        if cancel is not None:
            cancel()
        self.close()
