"""A concrete view producer for the common deferred buffer contract."""
import weakref
from infernux.compute_source import BufferSource


class _GraphBufferLease:
    def __init__(self, capture, engine):
        from infernux import compute
        self.capture = capture
        self.engine = weakref.ref(engine)
        self.shape, self.dtype, self.device = tuple(capture.shape), capture.dtype, 'gpu'
        self.storage_nbytes = capture.storage_byte_size
        self.host_identity = int(compute._native_compute_host().identity)

    @property
    def storage_claims(self):
        from infernux.compute import _native_storage_blocks
        known = self.capture.value if self.ready else self.capture.storage_buffer
        if known is not None:
            return dict(_native_storage_blocks(self.host_identity, known))
        key = ('future', self.host_identity, self.capture.view, self.capture.generation,
               self.capture.occurrence, self.capture.storage_claim)
        return {key: self.storage_nbytes}

    @property
    def transfer_key(self):
        import math
        import numpy as np
        dtype = np.dtype(self.dtype)
        native = self.capture.value if self.ready else self.capture.storage_buffer
        if native is not None:
            key = ('gpu', self.host_identity, native.resource_index, native.resource_generation)
        else:
            key = next(iter(self.storage_claims))
        return (*key, self.capture.byte_offset, math.prod(self.shape) * dtype.itemsize, dtype.name)

    @property
    def done(self):
        return self.capture.done

    @property
    def ready(self):
        return self.capture.ready

    @property
    def error(self):
        return self.capture.error

    @property
    def identity(self):
        return self.capture.view, self.capture.generation, self.capture.occurrence, self.capture.name

    def waits_for_render(self, engine):
        return not self.done and self.engine() is engine

    def acquire(self):
        from infernux import compute
        return compute.Buffer._from_native(self.capture.value, shape=self.shape,
                                           byte_offset=self.capture.byte_offset)

    def cancel(self):
        self.capture.cancel()


def capture_buffer(name, *, camera=None, scene_view=False):
    """Capture the next submitted execution of an already defined view export.

    The default is the primary Game camera. Specify a Camera or scene_view=True
    to select another view. Multiple calls before preparation share that exact
    next execution; calls after preparation wait for a subsequent execution.
    """
    from infernux.application import Application
    from infernux.components.builtin.camera import Camera
    if scene_view and camera is not None:
        raise ValueError('choose a Camera or the Scene view')
    if camera is not None and not isinstance(camera, Camera):
        raise TypeError('camera must be a Camera component')
    engine = Application._current_engine()
    if engine is None:
        raise RuntimeError('buffer captures require an active Engine')
    native = engine.get_native_engine()
    graph = (native.get_scene_render_graph() if scene_view else
             native.get_camera_render_graph(0 if camera is None else camera.component_id))
    if graph is None:
        raise RuntimeError('the selected view has not defined a render graph yet')
    return BufferSource(_GraphBufferLease(graph.capture_buffer(str(name)), engine))
