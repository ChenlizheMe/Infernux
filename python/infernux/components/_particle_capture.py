"""Particle observations backed by the common deferred numerical buffers."""
import math
import weakref

from infernux.compute_source import BufferSource


class _ParticleBufferLease:
    device = 'gpu'

    def __init__(self, output, engine_ref, host_identity, port, shape, dtype):
        self._output = output
        self._capture = output.capture
        self._engine = engine_ref
        self._host_identity = host_identity
        self._port = port
        self.shape = shape
        self.dtype = dtype
        self.storage_nbytes = math.prod(shape) * 4

    @property
    def storage_claims(self):
        from infernux.compute import _native_storage_blocks
        if self.ready:
            return dict(_native_storage_blocks(self._host_identity, self._output.buffer))
        return {('particle-capture', self._host_identity, self._capture.request_id, self._port): self.storage_nbytes}

    @property
    def transfer_key(self):
        return (*next(iter(self.storage_claims)), 0, self.storage_nbytes, self.dtype)

    @property
    def done(self):
        return self._capture.done

    @property
    def ready(self):
        return self._capture.ready

    @property
    def error(self):
        return self._capture.error

    @property
    def identity(self):
        return (*self._capture.identity, self._port)

    def waits_for_render(self, engine):
        return not self.done and self._engine() is engine

    def acquire(self):
        from infernux import compute
        if not self.ready:
            raise RuntimeError(self.error or 'GPU particle capture has not published a value')
        return compute.Buffer._from_native(self._output.buffer, shape=self.shape)

    def cancel(self):
        self._capture.cancel()


def particle_state_sources(ports, engine, rows, fields):
    from infernux import compute
    host_identity = int(compute._native_compute_host().identity)
    engine_ref = weakref.ref(engine)

    def source(index, components, dtype):
        return BufferSource(_ParticleBufferLease(ports[index], engine_ref, host_identity,
                                                 index, (rows, components), dtype))

    values = {
        name: source(port, components, dtype)
        for port, (name, components, dtype) in enumerate(fields)
    }
    return values, source(len(fields), 3, 'uint32')
