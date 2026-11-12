"""Resident compute values consumed by scene geometry."""


def _pose_range(value, stride):
    from infernux.compute import Buffer, _accepted_buffer_version

    if not isinstance(value, Buffer) or value._native is None:
        raise TypeError("PoseBuffer requires resident GPU buffers")
    value._require_open()
    if value._byte_offset % stride or value.nbytes % stride:
        raise ValueError(f"PoseBuffer requires complete {stride}-byte records")
    _accepted_buffer_version(value)
    return value._native, value._byte_offset // stride, value.nbytes // stride


class PoseBuffer:
    """Publish GPU models and final mesh bounds directly to renderers.

    Each model contains 16 float32 values in column order. Each world-space
    AABB contains min.xyz/padding followed by max.xyz/padding. Publish only
    after the producer's complete step has been accepted. Publication does
    not wait for the GPU or copy its results to the CPU.

    Bind with ``renderer.set_pose_buffer(source, element)``. Each camera keeps
    the value used by its previous accepted frame for motion vectors.
    """

    def __init__(self, models, bounds, *, count=None):
        from infernux.lib import _Infernux

        native_models, first_model, model_count = _pose_range(models, 64)
        native_bounds, first_bounds, bounds_count = _pose_range(bounds, 32)
        if count is None:
            count = model_count
        if count > model_count or count > bounds_count:
            raise ValueError("PoseBuffer count exceeds its model or bounds view")
        self._native = _Infernux._RenderPoseBuffer(native_models, native_bounds, count, first_model, first_bounds)

    @property
    def count(self):
        return self._native.count

    @property
    def generation(self):
        return self._native.generation

    def publish(self, models, bounds):
        """Freeze another complete value; existing frames retain their sample."""
        native_models, first_model, model_count = _pose_range(models, 64)
        native_bounds, first_bounds, bounds_count = _pose_range(bounds, 32)
        if self.count > model_count or self.count > bounds_count:
            raise ValueError("PoseBuffer count exceeds its model or bounds view")
        self._native.publish(native_models, native_bounds, first_model, first_bounds)
