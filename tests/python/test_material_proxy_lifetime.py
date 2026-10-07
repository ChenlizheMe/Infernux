"""Explicit disposal releases a proxy while native owners keep the resource."""
from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import pytest

import infernux as inx


def test_disposed_material_is_not_returned_by_native_conversion():
    material = inx.Material.create_unlit()
    native = material.native
    material.dispose()
    replacement = inx.Material.from_native(native)
    assert replacement is not material
    assert not replacement._disposed
    assert replacement.native is native
    assert inx.Material.from_native(native) is replacement


@pytest.mark.parametrize("access", ["native", "name", "set_float", "context"])
def test_disposed_material_access_reports_its_actual_lifetime(access):
    material = inx.Material.create_lit()
    material.dispose()
    material.dispose()  # Explicit retirement is idempotent.
    with pytest.raises(ReferenceError, match="disposed"):
        if access == "set_float":
            material.set_float("metallic", 0.1)
        elif access == "context":
            with material:
                pass
        else:
            getattr(material, access)


def test_disposal_releases_native_binding_without_changing_proxy_hash():
    material = inx.Material.create_unlit()
    native_ref = weakref.ref(material.native)
    original_hash = hash(material)
    owners = {material: "retired proxy"}
    material.dispose()
    gc.collect()
    assert native_ref() is None
    assert hash(material) == original_hash
    assert owners[material] == "retired proxy"


def test_renderer_keeps_material_after_python_context_exits(scene):
    owner = scene.create_game_object("Native material owner")
    renderer = inx.MeshRenderer._get_or_create_wrapper(owner.add_component("MeshRenderer"), owner)
    with inx.Material.create_unlit() as material:
        material.set_float("proxyLifetimeValue", 0.75)
        renderer.material = material
        assert renderer.material is material
    replacement = renderer.material
    assert replacement is not material
    assert not replacement._disposed
    assert replacement.get_float("proxyLifetimeValue") == pytest.approx(0.75)
    assert renderer.get_material(0) is replacement
    assert renderer.get_effective_material(0) is replacement
    renderer.material = None


def test_runtime_material_reference_does_not_keep_retired_proxy():
    material = inx.Material.create_unlit()
    reference = inx.MaterialRef(material)
    assert reference.resolve() is material
    material.dispose()
    assert reference.resolve() is None
    assert repr(material) == "<Material (disposed)>"


def test_registered_material_cache_and_reference_reacquire_live_proxy(engine, tmp_path, monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.engine.ui.project_file_ops import _new_material_document

    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    path = Path(database.assets_root) / tmp_path.name / "Lifetime.mat"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_new_material_document("Lifetime")), encoding="utf-8")
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported, imported.error
    material = AssetManager.load_by_guid(imported.guid, asset_type=inx.Material)
    reference = inx.MaterialRef(guid=imported.guid)
    assert reference.resolve() is material
    assert inx.Material.load(str(path)) is material
    material.dispose()
    replacement = AssetManager.load_by_guid(imported.guid, asset_type=inx.Material)
    assert replacement is not material
    assert replacement.guid == imported.guid
    assert not replacement._disposed
    assert reference.resolve() is replacement
    assert inx.Material.load(str(path)) is replacement
