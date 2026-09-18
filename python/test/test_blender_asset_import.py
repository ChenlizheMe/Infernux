"""The real asset database owns Blender conversion and binary publication."""
import os
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
        "root.location=(2,1,0); bpy.ops.mesh.primitive_cube_add(); "
        "cube=bpy.context.object; cube.name='ChildCube'; cube.parent=root; cube.location=(0,0,2); "
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
        assert mesh and mesh.vertex_count > 0 and mesh.submesh_count == 1
        assert mesh.name == source.stem
        assert source.read_bytes() == before
        assert database.last_refresh_worker_importer_count > 0
        staging = Path(database.project_root) / "Library/ModelImport"
        assert staging.is_dir() and not list(staging.iterdir())
        settings = read_mesh_import_settings(str(source))
        settings.scale_factor = 2
        result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert result, result.error
        assert result.guid == guid
        assert read_mesh_import_settings(str(source)).scale_factor == 2
        sidecar = Path(str(source) + ".meta").read_bytes()
        vertices = mesh.vertex_count
        source.write_bytes(b"invalid Blender source")
        result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
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
