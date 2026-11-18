from typing import Any, Mapping, Sequence
from infernux.components.builtin.rigidbody import Rigidbody
from infernux.compute import Buffer, BufferSource, Work

class Participant:
    """Component/World-owned persistent state and ordered author controls."""
    @property
    def ready(self) -> bool: ...
    def set_control(self, name: str, value: Any) -> None: ...
    def set_count(self, count: int) -> None: ...
    def capture_output(self, name: str, *, offset: int = 0, count: int | None = None) -> BufferSource: ...
    def close(self) -> None: ...

class PhysicsBuildPlan:
    """Supplied to InxComponent.build_physics after restoration, before Awake."""
    @property
    def step(self) -> Buffer:
        """f32[2]: substep dt, admitted fixed dt."""
        ...
    @property
    def indices(self) -> Buffer:
        """u32[6]: substep, substeps, iteration, iterations, count, stage."""
        ...
    def participant(self, *, initialize: Work, predict: Work | None = ..., prepare: Work | None = ...,
                    exchange: Work | None = ..., accept: Work | None = ...,
                    controls: Mapping[str, tuple[Buffer, Any]] | None = ..., count: int,
                    rigidbodies: Sequence[Rigidbody] = ...) -> Participant: ...
