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

    def begin_group(self):
        pass

    def end_group(self):
        pass

    def is_item_edited(self):
        return True

    def push_mono_font(self):
        pass

    def pop_font(self):
        pass

    def set_next_item_width(self, _width):
        pass

    def inspector_field_label(self, _label, _width):
        pass

    def get_cursor_pos_x(self):
        return 0.0

    def same_line(self, *_args):
        pass

    def float_slider(self, label, value, minimum, maximum, format):
        self.calls.append(("float_slider", label, value, minimum, maximum, format))
        return 0.75

    def int_slider(self, label, value, minimum, maximum):
        self.calls.append(("int_slider", label, value, minimum, maximum))
        return 3

    def drag_float(self, *_args):
        raise AssertionError("A ranged Float must use float_slider")

    def drag_int(self, *_args):
        raise AssertionError("A ranged Int must use int_slider")


def test_native_shader_metadata_owns_the_imported_schema_and_normalizes_hdr_values(tmp_path, monkeypatch):
    from infernux.core import asset_types

    shader = tmp_path / "structured.frag"
    shader.write_text('ShaderInfo { Name "Source/ShouldNotBeParsed" }\nvoid main() {}\n', encoding="utf-8")
    color = {
        "name": "base_color",
        "type": "Color",
        "default": [1.0, 0.5, 0.25, 1.0],
        "hdr": True,
        "range": [0.0, 1.0],
        "line": 7,
        "column": 9,
    }
    encoded = {"name": "emission_color", "type": "Color", "default": "[0.0, 0.0, 0.0, 0.0], HDR", "hdr": False}
    published = {
        "shader_id": "Imported/Structured",
        "shader_schema_format": "ShaderInfo",
        "properties": json.dumps([color, encoded]),
        "shader_hidden": True,
    }
    _write_meta(shader, {"shader_id": "Stale/Sidecar", "shader_hidden": False})
    monkeypatch.setattr(asset_types, "read_published_asset_metadata", lambda path: published)
    assert inspector_shader_utils.parse_shader_id(str(shader)) == "Imported/Structured"
    assert inspector_shader_utils.parse_shader_properties(str(shader)) == [
        color,
        {"name": "emission_color", "type": "Color", "default": [0.0, 0.0, 0.0, 0.0], "hdr": True},
    ]
    assert inspector_shader_utils.is_shader_hidden(str(shader))


def test_shader_schema_sync_drives_material_edits_and_removes_retired_range():
    data = {
        "properties": {
            "emission_color": {"type": 0, "value": "[0.0, 0.0, 0.0, 0.0], HDR"},
            "amount": {"type": 0, "value": 0.25},
            "steps": {"type": 4, "value": 1},
        }
    }
    schema = [
        {"name": "emission_color", "type": "Color", "default": "[0.1, 0.2, 0.3, 1.0], HDR", "hdr": False},
        {"name": "amount", "type": "Float", "default": 0.5, "hdr": False, "range": [0.0, 1.0]},
        {"name": "steps", "type": "Int", "default": 2, "hdr": False, "range": [0, 8]},
    ]
    inspector_shader_utils._apply_shader_props_to_mat(data, schema)
    assert data["properties"]["emission_color"] == {"type": 7, "value": [0.1, 0.2, 0.3, 1.0], "hdr": True}
    amount, steps = data["properties"]["amount"], data["properties"]["steps"]
    assert amount["value"] == 0.25 and amount["range"] == [0.0, 1.0]
    context = _RangePropertyContext()
    assert inspector_material.render_material_property(context, "amount", amount, 0, amount["value"], 80.0)
    assert inspector_material.render_material_property(context, "steps", steps, 4, steps["value"], 80.0)
    assert amount["value"] == 0.75 and steps["value"] == 3
    assert context.calls == [
        ("float_slider", "##mp_amount", 0.25, 0.0, 1.0, "%.3f"),
        ("int_slider", "##mp_steps", 1, 0, 8),
    ]
    schema[1].pop("range")
    inspector_shader_utils._apply_shader_props_to_mat(data, schema)
    assert data["properties"]["amount"]["value"] == 0.75
    assert "range" not in data["properties"]["amount"]


def test_shader_object_catalog_uses_author_suffixes_ids_and_material_domains(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from infernux.core.asset_reference_types import asset_type_registry
    from infernux.core.asset_types import ShaderAssetInfo, asset_category_from_extension
    from infernux.core.assets import AssetManager
    from infernux.core.shader import Shader
    from infernux.engine.interaction.object_fields import AssetReferenceCatalog

    paths = []
    metadata = {}
    for filename, base, shader_id, hidden in (
        ("paint.frag.py", "surface", "test/paint", False),
        ("post.frag.py", "fullscreen", "test/post", False),
        ("internal.frag.py", "surface", "test/internal", True),
        ("deform.vert.py", "vert", "test/deform", False),
    ):
        path = tmp_path / filename
        path.write_text(f'import infernux.shader as shader\nclass Effect(shader.{base}):\n'
                        f'    shader_id = {shader_id!r}\n    hidden = {hidden!r}\n', encoding="utf-8")
        paths.append(str(path))
        values = {"shader_id": shader_id, "shader_hidden": hidden,
                  "shader_domain": {"surface": "mesh", "fullscreen": "fullscreen", "vert": "vertex"}[base]}
        metadata[str(path)] = SimpleNamespace(has_key=values.__contains__,
            get_string=lambda key, values=values: values[key], get_bool=lambda key, values=values: values[key])
    monkeypatch.setattr(AssetManager, "_asset_database", SimpleNamespace(
        query_generation=1, get_all_asset_paths=lambda: paths,
        get_meta_by_path=lambda path: metadata.get(path)))
    monkeypatch.setattr(inspector_shader_utils, "_read_compiled_shader_metadata", lambda _: None)
    monkeypatch.setattr(Shader, "get_schema", lambda *_: (_ for _ in ()).throw(
        AssertionError("Picking shader metadata must not compile the program")))
    catalog = AssetReferenceCatalog()
    with monkeypatch.context() as no_io:
        no_io.setattr("builtins.open", lambda *_a, **_kw: pytest.fail("Picker opened an authoring file"))
        assert catalog.items("Shader.Fragment", "") == (("test/paint", paths[0]),)
        assert catalog.items("Shader.Vertex", "") == (("test/deform", paths[3]),)
        assert {name for name, _ in catalog.items("Shader", "")} == {"test/paint", "test/post", "test/deform"}
    assert not asset_type_registry.require("Shader.Fragment").incompatibility(paths[0])
    assert asset_type_registry.require("Shader.Vertex").incompatibility(paths[0])
    assert not asset_type_registry.require("Shader.Vertex").incompatibility(paths[3])
    assert AssetManager._type_from_extension(".frag.py") is Shader
    assert AssetManager._type_from_extension(".compute.py") is Shader
    assert AssetManager._type_from_extension(".py") is None
    assert asset_category_from_extension(".frag.py") == "shader"
    assert ShaderAssetInfo.from_path(paths[0]).shader_type == "fragment"
    assert ShaderAssetInfo.from_path(paths[3]).shader_type == "vertex"
    assert ShaderAssetInfo.from_path("solver.compute.py").shader_type == "compute"
    indexed = inspector_shader_utils._scan_shader_catalog(".frag", [str(tmp_path)])
    assert indexed["items"] == [("test/paint", "test/paint")]
    assert indexed["paths"]["test/internal"] == paths[2]
    assert indexed["paths"]["test/post"] == paths[1]


def test_shader_menu_groups_logical_ids_and_keeps_asset_identity():
    from infernux.engine.ui.igui import _picker_menu_tree

    items = (("infernux/lit", "physically/elsewhere/lit.frag.py"),
             ("infernux/sprite/lit", "assets/not_the_menu/sprite.frag.py"),
             ("my_game/surface/paint", "another/place.frag.py"))
    tree = _picker_menu_tree(items, inspector_shader_utils.shader_menu_folders)
    assert set(tree.folders) == {"infernux", "my_game"}
    assert tree.folders["infernux"].leaves == [("lit", items[0][1])]
    assert tree.folders["infernux"].folders["sprite"].leaves == [("lit", items[1][1])]
    assert tree.folders["my_game"].folders["surface"].leaves == [("paint", items[2][1])]
    assert _picker_menu_tree(items, inspector_shader_utils.shader_menu_folders) is tree
