from __future__ import annotations

from typing import Any, List, NotRequired, Optional, Tuple, TypedDict, Union
import numpy as np
import numpy.typing as npt
from Infernux.lib import PenetrationResult


class RaycastBatchProfile(TypedDict):
    input_validation_ms: float
    snapshot_sync_ms: float
    snapshot_lock_wait_ms: float
    dispatch_wall_ms: float
    jolt_query_cpu_ms: float
    broadphase_filter_lock_cpu_ms: float
    narrowphase_cpu_ms: float
    result_sort_filter_cpu_ms: float
    hit_publish_cpu_ms: float
    output_publish_ms: float
    broadphase_candidates: int
    narrowphase_hits: int
    published_hits: int


class RaycastBatchOutput(TypedDict):
    hit: npt.NDArray[np.uint8]
    point: npt.NDArray[np.float32]
    normal: npt.NDArray[np.float32]
    distance: npt.NDArray[np.float32]
    body_id: npt.NDArray[np.uint32]
    sub_shape_id: npt.NDArray[np.uint32]
    triangle_index: npt.NDArray[np.uint32]
    collider_id: npt.NDArray[np.uint64]
    game_object_id: npt.NDArray[np.uint64]
    query_generation: NotRequired[int]
    profile: NotRequired[RaycastBatchProfile]


class Physics:
    """Global physics system for raycasting and spatial queries."""

    @staticmethod
    def compute_penetration(collider_a: Any, position_a: Any, rotation_a: Any,
                            collider_b: Any, position_b: Any, rotation_b: Any) -> Optional[PenetrationResult]: ...

    @staticmethod
    def get_rigidbody_states(rigidbodies: List[Any], out: Optional[dict[str, Any]] = ...) -> dict[str, Any]: ...
    @staticmethod
    def get_rigidbody_states_and_box_states(
        rigidbodies: List[Any], state_out: dict[str, Any], box_out: dict[str, Any],
        *, query_triggers: bool = ...,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...
    @staticmethod
    def get_rigidbody_box_states(rigidbodies: List[Any], out: Optional[dict[str, Any]] = ...,
                                 *, query_triggers: bool = ...) -> dict[str, Any]: ...
    @staticmethod
    def query_rigidbody_box_states_in_bounds(
        minimum: Any, maximum: Any, out: dict[str, Any], layer_mask: int = ...,
        *, query_triggers: bool = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def query_rigidbody_states_and_box_states_in_bounds(
        minimum: Any, maximum: Any, state_out: dict[str, Any], box_out: dict[str, Any], layer_mask: int = ...,
        *, query_triggers: bool = ...,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    @staticmethod
    def apply_rigidbody_impulses(rigidbodies: List[Any], linear_impulses: Any,
                                  angular_impulses: Any) -> None: ...
    @staticmethod
    def set_contact_event_stream_enabled(enabled: bool = ..., *, include_triggers: bool = ...) -> None: ...
    @staticmethod
    def get_contact_events() -> dict[str, Any]: ...
    @staticmethod
    def set_contact_impulse_stream_enabled(enabled: bool = ...) -> None: ...
    @staticmethod
    def get_contact_impulses() -> dict[str, Any]: ...

    @staticmethod
    def query_rigidbodies_in_bounds(
        minimum: Any, maximum: Any, layer_mask: int = ..., query_triggers: bool = ...,
    ) -> List[Any]: ...

    @classmethod
    @property
    def body_count(cls) -> int:
        """Number of native physics bodies currently owned by the world."""
        ...

    @classmethod
    @property
    def query_generation(cls) -> int:
        """Monotonic token for the currently published physics query world."""
        ...

    @classmethod
    @property
    def gravity(cls) -> Any:
        """The global gravity vector applied to all rigidbodies."""
        ...
    @classmethod
    @gravity.setter
    def gravity(cls, value: Any) -> None: ...

    @staticmethod
    def get_gravity() -> Any:
        """Get the global gravity vector."""
        ...
    @staticmethod
    def set_gravity(value: Any) -> None:
        """Set the global gravity vector."""
        ...

    @staticmethod
    def raycast(
        origin: Any,
        direction: Any,
        max_distance: float = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> Optional[Any]:
        """Cast a ray and return the first hit, or None."""
        ...

    @staticmethod
    def raycast_screen(
        camera: Any,
        screen_position: Tuple[float, float],
        viewport_size: Tuple[float, float],
        max_distance: float = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> Optional[Any]:
        """Cast from a top-left-origin camera viewport pixel position."""
        ...

    @staticmethod
    def raycast_batch(
        origins: npt.NDArray[np.float32],
        directions: npt.NDArray[np.float32],
        out: RaycastBatchOutput,
        max_distance: float = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
        profile: bool = ...,
    ) -> RaycastBatchOutput: ...

    @staticmethod
    def raycast_all(
        origin: Any,
        direction: Any,
        max_distance: float = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> List[Any]:
        """Cast a ray and return all hits."""
        ...

    @staticmethod
    def overlap_sphere(
        center: Any,
        radius: float,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> List[Any]:
        """Find all colliders within a sphere."""
        ...

    @staticmethod
    def overlap_box(
        center: Any,
        half_extents: Any,
        orientation: Any = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> List[Any]:
        """Find all colliders within an oriented box."""
        ...

    @staticmethod
    def overlap_capsule(point0: Any, point1: Any, radius: float, layer_mask: int = ..., query_triggers: bool = ...) -> List[Any]: ...

    @staticmethod
    def sphere_cast(
        origin: Any,
        radius: float,
        direction: Any,
        max_distance: float = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> Optional[Any]:
        """Cast a sphere along a direction and return the first hit, or None."""
        ...

    @staticmethod
    def box_cast(
        center: Any,
        half_extents: Any,
        direction: Any,
        orientation: Any = ...,
        max_distance: float = ...,
        layer_mask: int = ...,
        query_triggers: bool = ...,
    ) -> Optional[Any]:
        """Cast a box along a direction and return the first hit, or None."""
        ...

    @staticmethod
    def capsule_cast(point0: Any, point1: Any, radius: float, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) -> Optional[Any]: ...

    @staticmethod
    def ignore_layer_collision(layer1: int, layer2: int, ignore: bool = ...) -> None:
        """Set whether collisions between two layers are ignored."""
        ...

    @staticmethod
    def get_ignore_layer_collision(layer1: int, layer2: int) -> bool:
        """Check if collisions between two layers are ignored."""
        ...

    @staticmethod
    def ignore_collision(collider1: Any, collider2: Any, ignore: bool = ...) -> None: ...

    @staticmethod
    def get_ignore_collision(collider1: Any, collider2: Any) -> bool: ...


__all__ = ["Physics"]
