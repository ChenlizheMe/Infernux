"""Source inventories and published rig/animation payloads are distinct."""
import base64
import json
import struct
import time
from pathlib import Path

import pytest

from Infernux.core.assets import AssetManager
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.lib import AssetRegistry


@pytest.mark.parametrize("source_kind", ["fbx", "animation_only", "skinned_gltf"])
@pytest.mark.parametrize("apply_mode", ["sync", "async"])
def test_rig_animation_apply_replaces_companion_without_losing_source_inventory(
    engine, tmp_path, monkeypatch, source_kind, apply_mode,
):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    root = Path(__file__).resolve().parents[2]
    if source_kind == "animation_only":
        source = folder / "EmptyAnimation.gltf"
        data = struct.pack("<8f", 0, 1, 0, 0, 0, 1, 0, 0)
        source.write_text(json.dumps({
            "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"name": "AnimatedEmpty"}],
            "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8}, {"buffer": 0, "byteOffset": 8, "byteLength": 24}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1]},
                          {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"}],
            "animations": [{"name": "Move", "samplers": [{"input": 0, "output": 1}],
                            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]}],
        }), encoding="utf-8")
    else:
        fixture = (root / "external/assimp/test/models/FBX/animation_with_skeleton.fbx" if source_kind == "fbx"
                   else root / "cpp/tests/fixtures/model_uv_basis.gltf")
        source = folder / fixture.name
        source.write_bytes(fixture.read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    guid = imported.guid
    try:
        mesh = registry.load_mesh(str(source))
        assert mesh.has_skinned_data
        original_bones = mesh.skinned_bone_count
        original_clips = mesh.skinned_animation_names
        original_vertices = mesh.vertex_count
        original_meta = database.get_meta_by_guid(guid).serialize_document()["metadata"]
        settings = read_mesh_import_settings(str(source))
        # Re-enable after each exclusion, including both flags changing together.
        for rig, animations in [("generic", False), ("none", False), ("none", True), ("generic", True)]:
            settings.rig_type, settings.import_animations = rig, animations
            if apply_mode == "sync":
                result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
            else:
                owner = AssetManager.begin_model_reimport(str(source), settings)
                deadline = time.monotonic() + 30
                while (result := AssetManager.poll_model_reimport(owner)) is None:
                    assert time.monotonic() < deadline
                    time.sleep(.002)
            assert result, result.error
            assert mesh.vertex_count == original_vertices
            assert mesh.skinned_animation_names == (original_clips if rig == "generic" and animations else [])
            assert mesh.skinned_bone_count == (original_bones if rig == "generic" else 0)
            assert mesh.has_skinned_data == (rig == "generic" and (bool(original_bones) or animations))
            metadata = database.get_meta_by_guid(guid).serialize_document()["metadata"]
            for key in ("bone_count", "bone_names_csv", "animation_count", "animation_names_csv"):
                assert metadata[key] == original_meta[key]
            saved = read_mesh_import_settings(str(source))
            assert (saved.rig_type, saved.import_animations) == (rig, animations)
            # Discard the live registry object and read the actual binary companion.
            registry.invalidate_asset(guid)
            mesh = registry.load_mesh(str(source))
            assert mesh.skinned_animation_names == (original_clips if rig == "generic" and animations else [])
            assert mesh.has_skinned_data == (rig == "generic" and (bool(original_bones) or animations))
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(source))


def test_model_pages_do_not_publish_or_discard_shared_drafts(monkeypatch):
    from Infernux.core.asset_types import MeshImportSettings
    from Infernux.engine.ui import asset_details_renderer as renderer
    renderer._ensure_categories()
    state = renderer._State()
    state.file_path = "model.fbx"
    state.category = "mesh"
    state.settings = MeshImportSettings(rig_type="none", import_animations=False)
    expected = state.settings.to_dict()
    monkeypatch.setattr(renderer, "_render_mesh_info", lambda *a: None)
    monkeypatch.setattr(renderer, "_render_model_materials", lambda *a: None)
    visited = []
    monkeypatch.setattr(renderer, "_render_import_fields", lambda *a, fields: visited.extend(f.key for f in fields))

    class Context:
        def __init__(self):
            self.active = "model"
            self.selected = []
        def begin_tab_bar(self, _): return True
        def end_tab_bar(self): pass
        def begin_tab_item(self, label, *, selected=False):
            if selected: self.selected.append(label)
            return label.endswith("_" + self.active)
        def end_tab_item(self): pass
        def record_semantic_item(self, *args): pass
        def text_wrapped(self, text): pass

    ctx = Context()
    for ctx.active in ("model", "rig", "animation", "materials", "model"):
        renderer._render_model_import_pages(ctx, None, state)
        assert state.settings.to_dict() == expected
    assert len(ctx.selected) == 1 and ctx.selected[0].endswith("_model")
    assert "rig_type" in visited and "import_animations" in visited
    assert "material_import_mode" in visited


def test_material_none_hides_source_operations_without_discarding_remaps(monkeypatch):
    from Infernux.core.asset_types import MeshImportSettings
    from Infernux.engine.ui import asset_details_renderer as renderer
    renderer._ensure_categories()
    state = renderer._State()
    state.file_path = "model.fbx"
    state.category = "mesh"
    state.settings = MeshImportSettings(material_import_mode="none", material_remaps={"material/A": "saved"})
    calls = []
    monkeypatch.setattr(renderer, "_render_model_materials", lambda *a: calls.append("materials"))
    monkeypatch.setattr(renderer, "_render_import_fields", lambda *a, **kw: calls.append("fields"))

    class Context:
        def begin_tab_bar(self, _): return True
        def end_tab_bar(self): pass
        def begin_tab_item(self, label, **kw): return label.endswith("_materials")
        def end_tab_item(self): pass
        def record_semantic_item(self, *a): pass
        def text_wrapped(self, text): pass

    renderer._render_model_import_pages(Context(), None, state)
    assert calls == ["fields"]
    assert state.settings.material_remaps == {"material/A": "saved"}
    state.settings.material_import_mode = "description"
    renderer._render_model_import_pages(Context(), None, state)
    assert calls == ["fields", "fields", "materials"]
