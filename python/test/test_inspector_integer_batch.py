"""Inspector integer batches preserve native unsigned masks without narrowing."""
from types import SimpleNamespace

import pytest

from Infernux import lib
from Infernux.components.fields import FieldType
from Infernux.engine.ui.inspector_utils import build_scalar_desc


@pytest.mark.parametrize("value,limits,slider", [
    (0xffffffff, (0, 0xffffffff), False),
    (0x80000000, (0, 0xffffffff), True),
    (-2147483648, None, False),
    (1 << 40, None, False),
])
def test_native_integer_batch_creation_refresh_and_render(engine, value, limits, slider):
    metadata = SimpleNamespace(field_type=FieldType.INT, range=limits,
                               slider=slider, drag_speed=None, multiline=False, tooltip="")
    descriptor = build_scalar_desc("##integer", "Integer", metadata, value)
    observed = {"renders": 0, "frames": 0, "error": None}

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            ctx.set_next_window_size(64., 64., 0)
            opened = ctx.begin_window("Integer###integer_contract", True, 0)
            try:
                if opened:
                    plan = ctx.create_property_batch_plan([descriptor])
                    assert plan.size == 1
                    assert ctx.render_property_batch_plan(plan, 12.) == {}
                    # A cached plan must also accept the same full-width value
                    # after its initial metadata/value decoding has completed.
                    assert ctx.render_property_batch_plan_values(plan, [value], 12.) == {}
                    observed["renders"] += 1
            except Exception as error:
                observed["error"] = error
            finally:
                ctx.end_window()

    def update(_delta):
        observed["frames"] += 1
        if observed["frames"] >= 4:
            engine.exit()

    probe = Probe()
    engine.register_gui_renderable("test.integer_batch", probe)
    try:
        engine.set_pre_scene_update_callback(update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.integer_batch")
    if observed["error"] is not None:
        raise observed["error"]
    assert observed["renders"] > 0
