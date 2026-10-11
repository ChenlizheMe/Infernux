"""SVG uses the same GUID/imported-GPU preview pipeline as raster textures."""
from pathlib import Path
import base64

import pytest

from infernux.lib import AssetRegistry, ResourceType, TextureLoader
from infernux.core.asset_types import (
    IMAGE_EXTENSIONS, TextureImportSettings, TextureCompression,
    read_texture_import_settings, write_texture_import_settings, read_meta_file,
)
from infernux.core.asset_reference_types import asset_type_registry
from infernux.core.assets import AssetManager

SVG = b'<svg viewBox="0 0 4 2"><rect width="4" height="2" fill="#ff0040" fill-opacity="0.5"/></svg>'
FIXTURE = Path(__file__).parents[1] / "native/fixtures/texture_vector.svg"


@pytest.mark.parametrize("size", [32, 128, 512])
@pytest.mark.parametrize("entry", ["file", "memory"])
def test_svg_loader_direct_rasterization(tmp_path, entry, size):
    path = tmp_path / "矢量.SVG"
    path.write_bytes(SVG)
    image = (TextureLoader.load_from_file(str(path), svg_max_size=size) if entry == "file"
             else TextureLoader.load_from_memory(SVG, svg_max_size=size))
    assert image.is_valid()
    assert (image.width, image.height) == (size, size // 2)
    pixels = image.get_pixels()
    assert len(pixels) == size * size * 2
    assert pixels[0] >= 253 and pixels[1] == 0 and 62 <= pixels[2] <= 66
    assert 126 <= pixels[3] <= 129


def test_svg_unlimited_and_raster_unlimited():
    settings = TextureImportSettings(max_size=0)
    assert TextureImportSettings.from_dict(settings.to_dict()).max_size == 0
    thin = b'<svg viewBox="0 0 8192 1"/>'
    image = TextureLoader.load_from_memory(thin)
    assert (image.width, image.height) == (8192, 1)
    # Unlike vectors, a raster source is not enlarged to fill the budget.
    image = TextureLoader.load_from_memory(b'P3\n1 1\n255\n255 0 0\n', svg_max_size=512)
    assert (image.width, image.height) == (1, 1)


def test_svg_embedded_images_but_no_host_file_reads(tmp_path):
    from PIL import Image
    path = tmp_path / "private.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(path)
    def decode(href):
        source = f'<svg xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 2 2"><image width="2" height="2" xlink:href="{href}"/></svg>'
        return TextureLoader.load_from_memory(source.encode(), svg_max_size=32).get_pixels()
    assert not any(decode(path.as_posix()))
    embedded = decode("data:image/png;base64," + base64.b64encode(path.read_bytes()).decode())
    assert embedded[:4] == bytes([255, 0, 0, 255])


def test_svg_import_reimport_guid_and_artifact(engine):
    assert ".svg" in IMAGE_EXTENSIONS
    registry = AssetRegistry.instance()
    database = registry.get_asset_database()
    path = Path(database.assets_root) / "vector-preview.svg"
    path.write_bytes(FIXTURE.read_bytes())
    result = database.import_asset(str(path))
    assert result, result.error
    guid = result.guid
    try:
        for name in ("Texture", "Texture.Sampled"):
            assert not asset_type_registry.require(name).incompatibility(str(path))
        settings = read_texture_import_settings(str(path))
        settings.max_size = 128
        settings.compression = TextureCompression.NONE
        settings.generate_mipmaps = True
        assert write_texture_import_settings(str(path), settings)
        result = AssetManager.reimport_asset(str(path), database=database)
        assert result and result.guid == guid, result.error
        assert read_texture_import_settings(str(path)).max_size == 128
        artifact = Path(database.get_runtime_artifact_path(guid, ResourceType.Texture))
        assert artifact.is_file() and artifact.stat().st_size > 128 * 64 * 4
        meta = read_meta_file(str(path))
        assert (meta["artifact_width"], meta["artifact_height"]) == (128, 64)
        assert (meta["width"], meta["height"]) == (128, 64)
        assert meta["source_container"] == "SVG" and not meta["is_binary"]
        # A malformed save must not replace the last good cooked publication.
        before = artifact.read_bytes()
        path.write_text("<svg><", encoding="utf-8")
        result = database.reimport_asset(str(path))
        assert not result and "SVG" in result.error
        assert database.get_guid_from_path(str(path)) == guid
        assert artifact.read_bytes() == before
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(path))
