"""Plugin documentation images stay faithful in the editor, never in a Player."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.asset_types import (
    TextureCompression, TextureFormat, read_meta_file,
    read_texture_import_settings, write_texture_import_settings,
)
from infernux.core.assets import AssetManager
from infernux.engine.project_context import is_editor_asset_path, is_plugin_page_path, using_project_root
from infernux.lib import AssetRegistry, ResourceType
from infernux.plugins.package import player_file_exported


def test_document_panel_requests_lossless_publication(tmp_path, monkeypatch):
    from infernux.engine.ui import asset_resource_preview as preview

    seen = []
    def query(*args, **kwargs):
        seen.append(kwargs)
        return 0, 321, 123
    monkeypatch.setattr(preview, "_resolve_native_engine", lambda panel:
                        SimpleNamespace(query_or_schedule_texture_preview=query))
    preview.render_document_image(None, None, str(tmp_path / "image.svg"), 720, 360)
    assert seen[0]["use_imported_texture"] is True
    assert seen[0]["texture_format"] == "auto"


@pytest.mark.parametrize("relative,expected", [
    ("Packages/demo/plugin_pages/image.png", True),
    ("packages/vendor/demo/runtime/docs/plugin_pages/nested/image.png", True),
    (r"Packages\vendor\demo\PLUGIN_PAGES\image.png", True),
    ("Packages/plugin_pages/image.png", True),
    ("Assets/plugin_pages/image.png", False),
    ("Other/Packages/demo/plugin_pages/image.png", False),
    ("Packages/demo/plugin_pages_backup/image.png", False),
    ("Packages/demo/plugin_pages.png", False),
    ("Packages/demo/plugin_pages", False),
])
def test_plugin_page_scope_and_inspector(tmp_path, relative, expected):
    from infernux.engine.ui.asset_details_renderer import _is_plugin_page_texture

    assert is_plugin_page_path(relative) is expected
    assert is_editor_asset_path(relative) is expected
    state = SimpleNamespace(category="texture", file_path=str(tmp_path / relative.replace("\\", "/")))
    with using_project_root(str(tmp_path)):
        assert _is_plugin_page_texture(state) is expected
        state.category = "mesh"
        assert not _is_plugin_page_texture(state)


@pytest.mark.parametrize("role", ["runtime", "content", "control", ""])
@pytest.mark.parametrize("logical", [
    "plugin_pages/image.png", "runtime/deep/plugin_pages/image.png",
    r"runtime\PLUGIN_PAGES\nested\data.bin",
])
def test_package_export_excludes_docs_despite_persisted_role(role, logical):
    assert not player_file_exported({"role": role}, logical)


def test_documentation_inspector_does_not_reinterpret_external_resources(tmp_path):
    from infernux.engine.ui.asset_details_renderer import _is_plugin_page_texture

    state = SimpleNamespace(category="texture", file_path=str(tmp_path / "Engine/resources/icons/camera.png"))
    with using_project_root(str(tmp_path / "Project")):
        assert not _is_plugin_page_texture(state)


@pytest.mark.parametrize("suffix,mode", [(".png", "RGBA"), (".jpg", "RGB"), (".png", "I;16")])
def test_plugin_page_native_import_preserves_dimensions_and_precision(engine, suffix, mode):
    from PIL import Image

    registry = AssetRegistry.instance()
    database = registry.get_asset_database()
    path = (Path(database.project_root) / "Packages/docs-test/runtime/nested/plugin_pages/images"
            / (mode.replace(";", "-") + suffix))
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (3001, 3), (19, 37, 211, 91) if mode == "RGBA" else
                      (19, 37, 211) if mode == "RGB" else 12345)
    image.save(path)
    original = path.read_bytes()
    imported = database.import_asset(str(path))
    assert imported, imported.error
    guid = imported.guid
    try:
        for attempt in range(2):
            metadata = read_meta_file(str(path))
            assert (metadata["artifact_width"], metadata["artifact_height"]) == (3001, 3)
            assert metadata["artifact_mip_count"] == 1
            assert metadata["artifact_format"] == ("rgba32_float" if mode == "I;16" else "rgba8_srgb")
            assert metadata["texture_compression"] == "none" and metadata["max_size"] == 0
            assert metadata["texture_format"] == "auto"
            assert not metadata["generate_mipmaps"]
            artifact = Path(database.get_runtime_artifact_path(guid, ResourceType.Texture))
            assert artifact.is_file()
            if mode == "RGBA":
                original_pixels = image.tobytes()
                assert artifact.read_bytes()[-len(original_pixels) - 8:-8] == original_pixels
            if mode == "I;16":
                import struct
                # Check the actual cooked pixels, not only the format label.
                channel = ((12345 / 65535 + .055) / 1.055) ** 2.4
                assert struct.unpack("<4f", artifact.read_bytes()[-24:-8]) == pytest.approx(
                    (channel, channel, channel, 1.0), abs=1e-7,
                )
            assert path.read_bytes() == original
            loaded = registry.load_texture_by_guid(guid)
            assert loaded is not None and (loaded.pixel_width, loaded.pixel_height) == (3001, 3)
            assert "bc" not in loaded.pixel_format.casefold() and loaded.mip_count == 1
            if attempt == 0:
                settings = read_texture_import_settings(str(path))
                settings.max_size = 32
                settings.compression = TextureCompression.BC1
                settings.format = TextureFormat.RGBA4444
                settings.generate_mipmaps = True
                assert write_texture_import_settings(str(path), settings)
                updated = AssetManager.reimport_asset(str(path), database=database)
                assert updated and updated.guid == guid, updated.error
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(path))


@pytest.mark.parametrize("batch", [False, True])
def test_moving_existing_texture_into_documentation_reimports_immediately(engine, batch):
    from PIL import Image

    registry = AssetRegistry.instance()
    database = registry.get_asset_database()
    old = Path(database.assets_root) / f"move-documentation-{batch}.png"
    new = Path(database.project_root) / "Packages/docs-test/runtime/plugin_pages" / old.name
    new.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGBA", (3001, 3), (11, 29, 255, 89)) as source:
        source.save(old)
    imported = database.import_asset(str(old))
    assert imported, imported.error
    guid = imported.guid
    try:
        assert read_meta_file(str(old))["artifact_width"] == 2048
        old.rename(new)
        if batch:
            moved = database.move_assets_batch([(str(old), str(new))])[0]
        else:
            moved = database.move_asset(str(old), str(new))
        assert moved and moved.guid == guid, moved.error
        metadata = read_meta_file(str(new))
        assert metadata["artifact_width"] == 3001
        assert metadata["artifact_format"] == "rgba8_srgb"
        assert metadata["texture_compression"] == "none" and metadata["max_size"] == 0
        assert metadata["artifact_mip_count"] == 1
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(new if new.exists() else old))


@pytest.mark.parametrize("attributes", ['width="321" height="123"', 'viewBox="0 0 321 123"'])
def test_documentation_svg_uses_its_authored_viewport(engine, attributes):
    registry = AssetRegistry.instance()
    database = registry.get_asset_database()
    path = Path(database.project_root) / "Packages/docs-test/plugin_pages/viewport.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg {attributes}><rect width="321" height="123" fill="red"/></svg>', encoding="utf-8")
    imported = database.import_asset(str(path))
    assert imported, imported.error
    try:
        metadata = read_meta_file(str(path))
        assert (metadata["artifact_width"], metadata["artifact_height"]) == (321, 123)
        assert metadata["artifact_format"] == "rgba8_srgb" and metadata["artifact_mip_count"] == 1
    finally:
        registry.invalidate_asset(imported.guid)
        database.delete_asset(str(path))
