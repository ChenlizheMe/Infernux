"""
SkinnedMeshRenderer — Python wrapper for the native SkinnedMeshRenderer component.

It reuses MeshRenderer's GUID-backed mesh/material API while exposing animated
model metadata and the active take selection used by SkeletalAnimator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from infernux.components.builtin.mesh_renderer import MeshRenderer
from infernux.components.builtin_component import CppProperty
from infernux.components.fields import FieldType


class SkinnedMeshRenderer(MeshRenderer):
    """Python facade for the native animated-model renderer component."""

    _cpp_type_name = "SkinnedMeshRenderer"
    _component_category_ = "Rendering"

    source_model_guid = CppProperty(
        "source_model_guid",
        FieldType.STRING,
        default="",
        readonly=True,
        visible_when=lambda _c: False,
        tooltip="GUID of the source animated model asset",
    )
    active_take_name = CppProperty.from_native("SkinnedMeshRenderer", "active_take_name")
    animation_source_guid = CppProperty(
        "animation_source_guid",
        FieldType.STRING,
        default="",
        readonly=True,
        visible_when=lambda _c: False,
        tooltip="GUID of the model that owns the active animation take",
    )

    def get_animation_take_names(self) -> List[str]:
        cpp = self._cpp_component
        if cpp is None:
            return []
        return list(cpp.get_animation_take_names())

    def set_source_model_guid(self, guid: str) -> None:
        cpp = self._cpp_component
        if cpp is not None and hasattr(cpp, "set_source_model_guid"):
            cpp.set_source_model_guid(guid or "")

    def get_skeleton_node_names(self) -> List[str]:
        """Return all unique skeleton node names, including unweighted attachment nodes."""
        return self._require_cpp_component().get_skeleton_node_names()

    def get_skeleton_node_world_matrices(self, node_names: Sequence[str]) -> NDArray[np.float32]:
        """Read the latest submitted pose as ``float32[N, 4, 4]`` matrices.

        Matrices use ``[node, row, column]`` and transform bone-local points in
        engine units into rendered world space, including import scale, the
        owner's Transform and a selected submesh's pivot. Read after animation
        updates to consume this frame's pose. Missing node names raise ValueError.
        """
        return self._require_cpp_component().get_skeleton_node_world_matrices(node_names)

    def submit_pose_stack(self, layers: Sequence[dict]) -> None:
        """Sample and combine weighted animation layers with exact bone masks.

        Each layer provides take_name, time, weight, loop and optional
        source_model_guid, bone_mask and additive. blend_group normalizes member
        weights per bone before group_weight applies a fade. Call after animator updates
        when composing their poses with independent locomotion or aiming.
        """
        self._require_cpp_component().submit_pose_stack(list(layers))

    def get_root_motion_delta(
        self, take_name: str, from_seconds: float, to_seconds: float,
        loop: bool = True, animation_source_guid: str = "",
    ) -> Tuple[object, object]:
        """Return (translation, rotation) relative to the root's from-time frame.

        Translation is in engine units. Apply it along the owner's current
        axes, then multiply its rotation by the returned rotation. Looping
        composes the full cycle motion, including turns; reverse intervals
        return the inverse motion. Non-looping times clamp to clip bounds.
        """
        return self._require_cpp_component().get_root_motion_delta(
            take_name, from_seconds, to_seconds, loop, animation_source_guid,
        )

    @property
    def animation_take_count(self) -> int:
        return len(self.get_animation_take_names())

    @property
    def has_animation_takes(self) -> bool:
        return self.animation_take_count > 0

    @property
    def runtime_animation_time(self) -> float:
        """Animation time submitted to the native renderer for the current pose."""
        return float(self._require_cpp_component().runtime_animation_time)

    @property
    def runtime_animation_normalized_time(self) -> float:
        """Normalized time submitted to the native renderer for the current pose."""
        return float(
            self._require_cpp_component().runtime_animation_normalized_time
        )
