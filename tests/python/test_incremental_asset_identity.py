"""Incremental import must not steal an already published asset identity."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", [
    "duplicate-text", "duplicate-texture", "duplicate-refresh", "deleted-owner",
    "replace-registered-guid", "replace-with-new-guid", "same-path", "missing-sidecar",
    "move-missing-sidecar", "move-mismatched-sidecar",
])
def test_incremental_asset_identity(tmp_path, case):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), case],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, case):
    import json
    import shutil
    import struct
    import uuid
    import zlib
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        suffix = ".png" if case == "duplicate-texture" else ".txt"
        first, second = assets / ("A" + suffix), assets / ("B" + suffix)
        if suffix == ".png":
            def chunk(kind, content):
                return (struct.pack(">I", len(content)) + kind + content
                        + struct.pack(">I", zlib.crc32(kind + content)))
            pixels = (b"\0" + bytes([255, 0, 0, 255]) * 2) * 2
            first.write_bytes(b"\x89PNG\r\n\x1a\n"
                              + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
                              + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
        else:
            first.write_text("first authored asset", encoding="utf-8")
        imported = database.import_asset(str(first))
        assert imported, imported.error
        original_guid = imported.guid
        metadata = database.get_meta_by_guid(original_guid).serialize_document()
        first_meta = Path(str(first) + ".meta")
        second_meta = Path(str(second) + ".meta")
        if case.startswith("move-"):
            document = json.loads(first_meta.read_bytes())
            first.rename(second)
            first_meta.unlink()
            generation = database.query_generation
            if case == "move-mismatched-sidecar":
                document["metadata"]["guid"]["value"] = uuid.uuid4().hex
                authored_bytes = json.dumps(document).encode("utf-8")
                second_meta.write_bytes(authored_bytes)
                results = database.move_assets_batch([(str(first), str(second))])
                assert len(results) == 1 and not results[0]
                assert "different identity" in results[0].error
                assert second_meta.read_bytes() == authored_bytes
                assert database.query_generation == generation
                assert database.get_guid_from_path(str(first)) == original_guid
                assert not database.get_guid_from_path(str(second))
            else:
                result = database.move_asset(str(first), str(second))
                assert result, result.error
                assert database.get_guid_from_path(str(second)) == original_guid
                assert not database.get_guid_from_path(str(first))
                moved = json.loads(second_meta.read_bytes())
                assert moved["metadata"]["guid"]["value"] == original_guid
                assert moved["metadata"]["file_path"]["value"] == "Assets/B.txt"
            return
        other_guid = None
        if case == "replace-registered-guid":
            second.write_text("other authored asset", encoding="utf-8")
            other = database.import_asset(str(second))
            assert other, other.error
            other_guid = other.guid
            shutil.copyfile(second_meta, first_meta)
            target = first
        elif case == "replace-with-new-guid":
            document = json.loads(first_meta.read_bytes())
            document["metadata"]["guid"]["value"] = uuid.uuid4().hex
            first_meta.write_text(json.dumps(document), encoding="utf-8")
            target = first
        elif case in {"same-path", "missing-sidecar"}:
            target = first
            if case == "missing-sidecar":
                first_meta.unlink()
        else:
            shutil.copyfile(first, second)
            shutil.copyfile(first_meta, second_meta)
            target = second
            if case == "deleted-owner":
                first.unlink()
        database.flush_derived_index()
        before_generation = database.query_generation
        before_index = Path(database.asset_index_path).read_bytes()
        before_files = {path: path.read_bytes() for path in (first_meta, second_meta) if path.exists()}
        artifacts = {path: path.read_bytes() for path in (project / "Library/Artifacts").rglob("*")
                     if path.is_file()}
        result, error = None, ""
        try:
            if case == "duplicate-refresh":
                database.refresh()
                result = True
            else:
                result = database.import_asset(str(target))
                error = result.error
        except RuntimeError as exc:
            error = str(exc)
        if case in {"same-path", "missing-sidecar"}:
            assert result, error
            assert result.guid == original_guid
            assert first_meta.is_file()
            assert database.get_guid_from_path(str(first)) == original_guid
            return
        assert not result, "conflicting import stole a published identity"
        assert error, "identity rejection must include a diagnostic"
        assert database.query_generation == before_generation
        assert database.get_guid_from_path(str(first)) == original_guid
        assert Path(database.get_path_from_guid(original_guid)) == first
        assert database.get_meta_by_guid(original_guid).serialize_document() == metadata
        assert database.get_guid_from_path(str(second)) == (other_guid or "")
        for path, content in before_files.items():
            assert path.read_bytes() == content, path
        for path, content in artifacts.items():
            assert path.read_bytes() == content, path
        database.flush_derived_index()
        assert Path(database.asset_index_path).read_bytes() == before_index
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
