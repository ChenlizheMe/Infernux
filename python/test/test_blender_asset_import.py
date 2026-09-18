"""The real asset database owns Blender conversion and binary publication."""
import os
import json
from pathlib import Path
import subprocess
import time

import pytest

from Infernux.core.assets import AssetManager
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.engine.model_import.toolchain import export_script
from Infernux.lib import AssetRegistry


def test_blender_tool_requires_explicit_absolute_configuration(engine, tmp_path):
    database = engine.get_asset_database()
    with pytest.raises(ValueError):
        database.configure_blender_import("blender", export_script())
    with pytest.raises(ValueError):
        database.configure_blender_import("", export_script())
    database.configure_blender_import("", "")
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "MissingTool.blend"
    source.write_bytes(b"BLENDER-v410")
    try:
        result = database.import_asset(str(source))
        assert not result and "not configured" in result.error
        assert not (Path(database.project_root) / "Library/ModelImport").exists()
    finally:
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(str(source) + ".meta").unlink(missing_ok=True)


@pytest.mark.skipif(not os.environ.get("INFERNUX_TEST_BLENDER"), reason="requires the Blender 5.2 authoring tool")
def test_modern_blend_worker_import_reimport_and_failed_publication(engine, tmp_path, monkeypatch):
    tool = os.environ["INFERNUX_TEST_BLENDER"]
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "嵌套 模型.BLEND"
    # This source is authored by Blender itself, not a renamed glTF fixture.
    script = (
        "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); "
        "root=bpy.data.objects.new('Assembly',None); bpy.context.collection.objects.link(root); "
        "root.location=(2,1,0); "
        "red=bpy.data.materials.new('Red'); red.diffuse_color=(0.8,0.1,0.1,1); "
        "red.use_nodes=True; image=bpy.data.images.new('Embedded Color',width=2,height=2); "
        "image.pixels=[1,0.05,0.05,1, 0.8,0.1,0.1,1, 0.6,0.15,0.15,1, 0.4,0.2,0.2,1]; "
        "nodes=red.node_tree.nodes; links=red.node_tree.links; bsdf=nodes.get('Principled BSDF'); "
        "tex=nodes.new('ShaderNodeTexImage'); tex.image=image; links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); "
        "blue=bpy.data.materials.new('Blue'); blue.diffuse_color=(0.1,0.2,0.8,1); "
        "bpy.ops.mesh.primitive_cube_add(); cube=bpy.context.object; "
        "cube.name='ChildCube'; cube.parent=root; cube.location=(0,0,2); cube.data.materials.append(red); "
        "bpy.ops.mesh.primitive_cube_add(); second=bpy.context.object; "
        "second.name='SecondCube'; second.parent=root; second.location=(2,0,2); second.data.materials.append(blue); "
        "second.keyframe_insert(data_path='location',frame=1); second.location=(2,1,2); "
        "second.keyframe_insert(data_path='location',frame=24); "
        "bpy.context.scene.frame_end=24; "
        "bpy.ops.wm.save_as_mainfile(filepath=" + repr(str(source)) + ")"
    )
    subprocess.run([tool, "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                    "--python-expr", script], check=True, capture_output=True, timeout=60,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    before = source.read_bytes()
    invalid_source = folder / "ConcurrentInvalid.blend"
    invalid_source.write_bytes(b"not a Blender file")
    database.configure_blender_import(tool, export_script())
    guid = None
    try:
        database.begin_refresh()
        ticks = 0
        deadline = time.monotonic() + 150
        while not database.try_commit_refresh():
            ticks += 1
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert ticks > 1  # Converted on the native worker, not inside BeginRefresh.
        guid = database.get_guid_from_path(str(source))
        assert guid
        mesh = registry.load_mesh(str(source))
        assert mesh and mesh.vertex_count > 0 and mesh.submesh_count >= 2
        assert mesh.name == source.stem
        manifest = json.loads(database.get_meta_by_guid(guid).get_string("model_meshes"))
        assert {entry["name"] for entry in manifest} >= {"ChildCube", "SecondCube"}
        assert all(entry["subresource_id"] for entry in manifest)
        textures = json.loads(database.get_meta_by_guid(guid).get_string("model_textures"))
        assert textures and any(record["name"] == "Embedded Color" for record in textures)
        nodes = mesh.get_model_nodes()
        assert {node["name"] for node in nodes} >= {"Assembly", "ChildCube", "SecondCube"}
        assert database.get_meta_by_guid(guid).get_int("animation_count") > 0
        assert source.read_bytes() == before
        assert database.last_refresh_worker_importer_count > 0
        staging = Path(database.project_root) / "Library/ModelImport"
        assert staging.is_dir() and not list(staging.iterdir())
        settings = read_mesh_import_settings(str(source))
        settings.scale_factor = 2
        started = time.monotonic()
        database.begin_model_reimport(str(source), settings.to_dict())
        assert time.monotonic() - started < 1.0
        assert read_mesh_import_settings(str(source)).scale_factor == 1
        apply_ticks = 0
        deadline = time.monotonic() + 150
        while (result := AssetManager.poll_model_reimport(database)) is None:
            apply_ticks += 1
            assert database.get_guid_from_path(str(source)) == guid
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert apply_ticks > 1  # Blender was running while owner queries remained available.
        assert result, result.error
        assert result.guid == guid
        assert read_mesh_import_settings(str(source)).scale_factor == 2
        sidecar = Path(str(source) + ".meta").read_bytes()
        vertices = mesh.vertex_count
        source.write_bytes(b"invalid Blender source")
        database.begin_model_reimport(str(source), settings.to_dict())
        deadline = time.monotonic() + 150
        while (result := AssetManager.poll_model_reimport(database)) is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert not result and "Blender import failed" in result.error
        assert Path(str(source) + ".meta").read_bytes() == sidecar
        assert mesh.vertex_count == vertices
        assert not list(staging.iterdir())
        # Binary reload must remain usable without the converter or source.
        database.configure_blender_import("", "")
        registry.invalidate_asset(guid)
        assert registry.load_mesh(str(source)).vertex_count == vertices
    finally:
        if guid:
            registry.invalidate_asset(guid)
        database.configure_blender_import("", "")
        database.delete_asset(str(source))
        database.delete_asset(str(invalid_source))
        for authored in (source, invalid_source):
            authored.unlink(missing_ok=True)
            Path(str(authored) + ".meta").unlink(missing_ok=True)
