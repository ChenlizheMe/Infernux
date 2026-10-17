from __future__ import annotations

import json

from infernux.engine.ui import inspector_shader_utils
from infernux.engine.ui import inspector_material


def _write_meta(path, metadata):
    document = {
        "metadata": {
            key: {
                "type": "bool" if isinstance(value, bool) else "string",
                "value": value,
            }
            for key, value in metadata.items()
        },
    }
    path.with_name(path.name + ".meta").write_text(json.dumps(document), encoding="utf-8")


class _StyleScopeContext:
    def __init__(self):
        self.calls = []

    def push_style_var_vec2(self, *_args):
        self.calls.append("push")

    def pop_style_var(self, count):
        self.calls.append(("pop", count))

    def begin_disabled(self, _disabled):
        self.calls.append("begin_disabled")

    def end_disabled(self):
        self.calls.append("end_disabled")


class _RangePropertyContext:
    def __init__(self):
        self.calls = []

    def align_text_to_frame_padding(self):
        pass

    def label(self, _label):
        pass

    def same_line(self, _width):
        pass

    def set_next_item_width(self, _width):
        pass

    def float_slider(self, label, value, minimum, maximum):
        self.calls.append(("float_slider", label, value, minimum, maximum))
        return 0.75

    def int_slider(self, label, value, minimum, maximum):
        self.calls.append(("int_slider", label, value, minimum, maximum))
        return 3

    def drag_float(self, *_args):
        raise AssertionError("A ranged Float must use float_slider")

    def drag_int(self, *_args):
        raise AssertionError("A ranged Int must use int_slider")


def test_native_shader_metadata_owns_the_imported_schema_and_normalizes_hdr_values(tmp_path):
    shader = tmp_path / "structured.frag"
    shader.write_text('ShaderInfo { Name "Source/ShouldNotBeParsed" }\nvoid main() {}\n', encoding="utf-8")
    color = {
        "name": "baseColor",
        "type": "Color",
        "default": [1.0, 0.5, 0.25, 1.0],
        "hdr": True,
        "range": [0.0, 1.0],
        "line": 7,
        "column": 9,
    }
    encoded = {"name": "emissionColor", "type": "Color", "default": "[0.0, 0.0, 0.0, 0.0], HDR", "hdr": False}
    _write_meta(
        shader,
        {
            "shader_id": "Imported/Structured",
            "shader_schema_format": "ShaderInfo",
            "properties": json.dumps([color, encoded]),
            "shader_hidden": True,
        },
    )
    assert inspector_shader_utils.parse_shader_id(str(shader)) == "Imported/Structured"
    assert inspector_shader_utils.parse_shader_properties(str(shader)) == [
        color,
        {"name": "emissionColor", "type": "Color", "default": [0.0, 0.0, 0.0, 0.0], "hdr": True},
    ]
    assert inspector_shader_utils.is_shader_hidden(str(shader))


def test_shader_schema_sync_drives_material_edits_and_removes_retired_range():
    data = {
        "properties": {
            "emissionColor": {"type": 0, "value": "[0.0, 0.0, 0.0, 0.0], HDR"},
            "amount": {"type": 0, "value": 0.25},
            "steps": {"type": 4, "value": 1},
        }
    }
    schema = [
        {"name": "emissionColor", "type": "Color", "default": "[0.1, 0.2, 0.3, 1.0], HDR", "hdr": False},
        {"name": "amount", "type": "Float", "default": 0.5, "hdr": False, "range": [0.0, 1.0]},
        {"name": "steps", "type": "Int", "default": 2, "hdr": False, "range": [0, 8]},
    ]
    inspector_shader_utils._apply_shader_props_to_mat(data, schema)
    assert data["properties"]["emissionColor"] == {"type": 7, "value": [0.1, 0.2, 0.3, 1.0], "hdr": True}
    amount, steps = data["properties"]["amount"], data["properties"]["steps"]
    assert amount["value"] == 0.25 and amount["range"] == [0.0, 1.0]
    context = _RangePropertyContext()
    assert inspector_material.render_material_property(context, "amount", amount, 0, amount["value"], 80.0)
    assert inspector_material.render_material_property(context, "steps", steps, 4, steps["value"], 80.0)
    assert amount["value"] == 0.75 and steps["value"] == 3
    assert context.calls == [("float_slider", "##mp_amount", 0.25, 0.0, 1.0), ("int_slider", "##mp_steps", 1, 0, 8)]
    schema[1].pop("range")
    inspector_shader_utils._apply_shader_props_to_mat(data, schema)
    assert data["properties"]["amount"]["value"] == 0.75
    assert "range" not in data["properties"]["amount"]
