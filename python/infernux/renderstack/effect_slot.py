"""Structured RenderStack stage slots stored by the component serializer."""

from __future__ import annotations

import uuid
from weakref import WeakSet

from infernux.components.serializable_object import SerializableObject
from infernux.components.fields import serialized_field
from infernux.core.asset_ref import RenderEffectRef


class EffectSlot(SerializableObject):
    """One ordered Effect asset mount in a pipeline-defined stage."""

    slot_id: str = serialized_field(default="", hidden=True)
    stage_id: str = serialized_field(default="", hidden=True)
    effect: RenderEffectRef = serialized_field(default=None, asset_type="RenderEffect")
    enabled: bool = serialized_field(default=True)

    def __setattr__(self, name: str, value) -> None:
        values = vars(self)
        previous = values.get(name)
        super().__setattr__(name, value)
        if name in ("enabled", "effect", "stage_id", "slot_id") and previous != values[name]:
            for stack in values.get("_graph_owners", ()):
                stack.invalidate_graph()

    def _observe(self, stack):
        owners = vars(self).get("_graph_owners")
        if owners is None:
            owners = vars(self)["_graph_owners"] = WeakSet()
        owners.add(stack)

    def _unobserve(self, stack):
        vars(self)["_graph_owners"].discard(stack)

    def __init__(
        self,
        *,
        slot_id: str = "",
        stage_id: str = "",
        effect=None,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            slot_id=slot_id or uuid.uuid4().hex,
            stage_id=stage_id,
            effect=effect,
            enabled=enabled,
        )

    @property
    def effect_ref(self) -> RenderEffectRef:
        """Return the raw asset reference without resolving it."""
        from infernux.components.fields import get_raw_field_value

        return get_raw_field_value(self, "effect")
