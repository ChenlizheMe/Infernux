"""RenderStack snapshot/restore and undo commands for the render pipeline."""

from __future__ import annotations

import json as _json
from typing import Any, Optional

from Infernux.debug import Debug
from Infernux.engine.undo._base import UndoCommand
from Infernux.engine.undo._property_commands import SetPropertyCommand


# -- Snapshot/restore --

def _serialize_simple(val: Any) -> Any:
    import enum
    if isinstance(val, enum.Enum):
        return val.value
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_simple(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialize_simple(v) for k, v in val.items()}
    if hasattr(val, 'x') and hasattr(val, 'y'):
        if hasattr(val, 'w'):
            return [val.x, val.y, val.z, val.w]
        if hasattr(val, 'z'):
            return [val.x, val.y, val.z]
        return [val.x, val.y]
    return str(val)


def snapshot_renderstack(stack: Any) -> str:
    from Infernux.components.serialized_field import get_serialized_fields

    data: dict = {
        "pipeline_class_name": stack.pipeline_class_name or "",
        "pipeline_params": {},
        "pass_entries": [],
    }
    pipeline = stack.pipeline
    if pipeline:
        for name, meta in get_serialized_fields(pipeline.__class__).items():
            val = getattr(pipeline, name, meta.default)
            data["pipeline_params"][name] = _serialize_simple(val)

    for entry in stack.pass_entries:
        ed: dict = {
            "class": type(entry.render_pass).__name__,
            "name": entry.render_pass.name,
            "enabled": entry.enabled,
            "order": entry.order,
        }
        if hasattr(entry.render_pass, 'get_params_dict'):
            ed["params"] = entry.render_pass.get_params_dict()
        data["pass_entries"].append(ed)

    return _json.dumps(data, sort_keys=True)


def restore_renderstack(stack: Any, json_str: str) -> None:
    from Infernux.components.serialized_field import get_serialized_fields

    data = _json.loads(json_str)

    new_pipeline_name = data.get("pipeline_class_name", "")
    if (stack.pipeline_class_name or "") != new_pipeline_name:
        stack.set_pipeline(new_pipeline_name)

    pipeline = stack.pipeline
    if pipeline and "pipeline_params" in data:
        for name, val in data["pipeline_params"].items():
            try:
                setattr(pipeline, name, val)
            except (AttributeError, TypeError) as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")

    current_names = [e.render_pass.name for e in list(stack.pass_entries)]
    for name in current_names:
        stack.remove_pass(name)

    from Infernux.renderstack.discovery import discover_passes
    from Infernux.renderstack.fullscreen_effect import FullScreenEffect

    all_passes = discover_passes()
    for ed in data.get("pass_entries", []):
        cls_name = ed.get("class", "")
        pass_name = ed.get("name", "")
        cls = all_passes.get(pass_name)
        if cls is None:
            for pcls in all_passes.values():
                if pcls.__name__ == cls_name:
                    cls = pcls
                    break
        if cls is None:
            continue
        inst = cls()
        if isinstance(inst, FullScreenEffect) and "params" in ed:
            inst.set_params_dict(ed["params"])
        inst.enabled = ed.get("enabled", True)
        stack.add_pass(inst)
        stack.set_pass_enabled(pass_name, ed.get("enabled", True))
        stack.reorder_pass(pass_name, ed.get("order", 0))

    stack.invalidate_graph()


# -- RenderStack undo commands --

class RenderStackFieldCommand(SetPropertyCommand):
    def __init__(self, stack: Any, target: Any, field_name: str,
                 old_value: Any, new_value: Any, description: str = ""):
        super().__init__(target, field_name, old_value, new_value,
                         description or f"Set {field_name}")
        self._stack = stack

    def execute(self) -> None:
        super().execute()
        self._stack.invalidate_graph()

    def undo(self) -> None:
        super().undo()
        self._stack.invalidate_graph()

    def redo(self) -> None:
        super().redo()
        self._stack.invalidate_graph()


class RenderStackSetPipelineCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, old_pipeline: str, new_pipeline: str,
                 description: str = "Set Render Pipeline"):
        super().__init__(description)
        self._stack = stack
        self._old_pipeline = old_pipeline
        self._new_pipeline = new_pipeline

    def execute(self) -> None:
        self._stack.set_pipeline(self._new_pipeline)

    def undo(self) -> None:
        self._stack.set_pipeline(self._old_pipeline)

    def redo(self) -> None:
        self._stack.set_pipeline(self._new_pipeline)


class RenderStackAddPassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, effect_cls: type,
                 description: str = "Add Effect"):
        super().__init__(description)
        self._stack = stack
        self._effect_cls = effect_cls
        self._pass_name: str = getattr(effect_cls, "name", effect_cls.__name__)
        self._snapshot: Optional[str] = None

    def execute(self) -> None:
        self._snapshot = snapshot_renderstack(self._stack)
        inst = self._effect_cls()
        self._stack.add_pass(inst)
        self._pass_name = inst.name

    def undo(self) -> None:
        if self._snapshot:
            restore_renderstack(self._stack, self._snapshot)
        else:
            self._stack.remove_pass(self._pass_name)

    def redo(self) -> None:
        self.execute()


class RenderStackMovePassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, old_orders: dict, new_orders: dict,
                 description: str = "Reorder Effect"):
        super().__init__(description)
        self._stack = stack
        self._old_orders = dict(old_orders)
        self._new_orders = dict(new_orders)

    def execute(self) -> None:
        self._apply(self._new_orders)

    def undo(self) -> None:
        self._apply(self._old_orders)

    def redo(self) -> None:
        self._apply(self._new_orders)

    def _apply(self, orders):
        for entry in self._stack.pass_entries:
            name = entry.render_pass.name
            if name in orders:
                entry.order = int(orders[name])
        self._stack.invalidate_graph()


class RenderStackTogglePassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, pass_name: str, old_enabled: bool,
                 new_enabled: bool, description: str = "Toggle Effect"):
        super().__init__(description)
        self._stack = stack
        self._pass_name = pass_name
        self._old_enabled = bool(old_enabled)
        self._new_enabled = bool(new_enabled)

    def execute(self) -> None:
        self._stack.set_pass_enabled(self._pass_name, self._new_enabled)

    def undo(self) -> None:
        self._stack.set_pass_enabled(self._pass_name, self._old_enabled)

    def redo(self) -> None:
        self._stack.set_pass_enabled(self._pass_name, self._new_enabled)


class RenderStackRemovePassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, pass_name: str,
                 description: str = "Remove Effect"):
        super().__init__(description)
        self._stack = stack
        self._pass_name = pass_name
        self._snapshot: Optional[str] = None

    def execute(self) -> None:
        self._snapshot = snapshot_renderstack(self._stack)
        self._stack.remove_pass(self._pass_name)

    def undo(self) -> None:
        if self._snapshot:
            restore_renderstack(self._stack, self._snapshot)

    def redo(self) -> None:
        self._stack.remove_pass(self._pass_name)
