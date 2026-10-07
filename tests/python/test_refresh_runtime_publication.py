"""Full catalog refresh must publish the same asset state to runtime consumers."""
from pathlib import Path
import os
import struct
import subprocess
import sys
import time
import zlib

import pytest


@pytest.mark.parametrize("case", [
    "modified", "deleted", "moved", "unchanged", "failed", "restored",
    "ticket-modified", "ticket-deleted", "ticket-preview", "ticket-unchanged",
    "ticket-moved", "ticket-preview-residency", "settings", "failed-recovered",
])
def test_refresh_runtime_publication(tmp_path, case):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), case],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def write_png(path, size):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + bytes([255, 0, 0, 255]) * size) * size))
                     + chunk(b"IEND", b""))


def exercise(project, case):
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import AssetRegistry, LogLevel, RuntimeMode

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        registry = AssetRegistry.instance()
        source = assets / "image.png"
        write_png(source, 2)
        imported = database.import_asset(str(source))
        assert imported, imported.error
        guid = imported.guid
        meta = Path(str(source) + ".meta")
        sidecar = meta.read_bytes()
        other = assets / "untouched.png"
        write_png(other, 4)
        other_guid = database.import_asset(str(other)).guid
        other_texture = registry.load_texture_by_guid(other_guid)
        other_version = registry.get_asset_version(other_guid)
        ticket = texture = None
        if case.startswith("ticket-"):
            ticket = registry.begin_load_texture_by_guid(guid)
            deadline = time.monotonic() + 15
            while not ticket.complete and time.monotonic() < deadline:
                time.sleep(.001)
            assert ticket.complete
        else:
            texture = registry.load_texture_by_guid(guid)
            assert texture.pixel_width == 2
        version = registry.get_asset_version(guid)
        if case == "ticket-preview-residency":
            # Another consumer can populate and then release the same unchanged
            # content while a preview ticket is waiting for owner publication.
            registry.load_texture_by_guid(guid)
            registry.cpu_budget_bytes = 1
            assert not registry.is_loaded(guid)
        if case in ("deleted", "restored", "ticket-deleted"):
            source.unlink()
            meta.unlink()
        elif case in ("moved", "ticket-moved"):
            destination = assets / "Renamed.png"
            source.rename(destination)
            meta.rename(str(destination) + ".meta")
            source = destination
        elif case in ("failed", "failed-recovered"):
            source.write_bytes(b"not a valid png")
        elif case == "settings":
            import json
            settings = json.loads(sidecar)
            settings["metadata"]["srgb"]["value"] = not texture.srgb
            meta.write_text(json.dumps(settings), encoding="utf-8")
        elif case not in ("unchanged", "ticket-unchanged", "ticket-preview-residency"):
            write_png(source, 3)
        database.refresh()
        assert registry.get_texture_asset(other_guid) is other_texture
        assert other_texture.pixel_width == 4
        assert registry.get_asset_version(other_guid) == other_version, "untouched asset was republished"
        if case in ("deleted", "restored"):
            assert not database.get_path_from_guid(guid)
            assert not registry.is_loaded(guid), "deleted asset remained cached"
            assert registry.load_texture_by_guid(guid) is None
            if case == "restored":
                write_png(source, 3)
                meta.write_bytes(sidecar)
                database.refresh()
                assert database.get_guid_from_path(str(source)) == guid
                assert registry.load_texture_by_guid(guid).pixel_width == 3
        elif ticket is not None:
            if case in ("ticket-unchanged", "ticket-preview-residency"):
                assert registry.try_commit_asset_load(ticket, allow_stale_if_unloaded=True)
                assert registry.load_texture_by_guid(guid).pixel_width == 2
            else:
                with pytest.raises(RuntimeError, match="stale"):
                    registry.try_commit_asset_load(ticket, allow_stale_if_unloaded=case == "ticket-preview")
                current = registry.load_texture_by_guid(guid)
                if case == "ticket-deleted":
                    assert current is None
                else:
                    assert current.pixel_width == (2 if case == "ticket-moved" else 3)
                    assert current.file_path.replace("\\", "/") == source.as_posix()
        else:
            current = registry.load_texture_by_guid(guid)
            assert current is texture, "refresh replaced a live resource object"
            assert current.file_path.replace("\\", "/") == source.as_posix()
            assert current.pixel_width == (3 if case == "modified" else 2)
            if case in ("unchanged", "failed", "failed-recovered", "moved"):
                assert registry.get_asset_version(guid) == version
            else:
                assert registry.get_asset_version(guid) > version
            if case == "settings":
                assert current.srgb == settings["metadata"]["srgb"]["value"]
            if case == "failed-recovered":
                write_png(source, 3)
                database.refresh()
                assert registry.load_texture_by_guid(guid) is texture
                assert texture.pixel_width == 3
                assert registry.get_asset_version(guid) > version
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
