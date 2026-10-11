from __future__ import annotations

from typing import Any

from infernux.components.serializable_object import SerializableObject
from infernux.core.asset_ref import RenderEffectRef
from infernux.renderstack.render_effect import RenderEffect

class EffectSlot(SerializableObject):
    slot_id: str
    stage_id: str
    effect: RenderEffect | None
    enabled: bool
    def __init__(self, *, slot_id: str = ..., stage_id: str = ..., effect: Any = ..., enabled: bool = ...) -> None: ...
    @property
    def effect_ref(self) -> RenderEffectRef: ...
