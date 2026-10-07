"""An import candidate cannot overwrite metadata edited after its capture."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("change", ["none", "write", "replace", "delete", "same-stat", "create", "source"])
def test_async_refresh_preserves_external_metadata(tmp_path, change):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), change],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, change):
    import json
    import struct
    import time
    import zlib
    from infernux.core.assets import AssetManager
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode

    def png(path, size):
        def chunk(kind, content):
            return (struct.pack('>I', len(content)) + kind + content
                    + struct.pack('>I', zlib.crc32(kind + content)))
        header = struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)
        pixels = (b'\0' + bytes([255, 0, 0, 255]) * size) * size
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
                         + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))

    assets = project / 'Assets'
    assets.mkdir()
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = AssetManager.require_asset_database()
        source = assets / 'image.png'
        png(source, 2)
        imported = AssetManager.import_asset(str(source))
        assert imported and imported.guid
        peer = assets / 'peer.png'
        png(peer, 2)
        peer_imported = AssetManager.import_asset(str(peer))
        assert peer_imported and peer_imported.guid
        peer_sidecar = Path(str(peer) + '.meta')
        peer_metadata = peer_sidecar.read_bytes()
        peer_artifact = project / f'Library/Artifacts/Texture/{peer_imported.guid}.inxtex'
        peer_bytes = peer_artifact.read_bytes()
        database.flush_derived_index()
        generation = database.query_generation
        old_metadata = database.get_meta_by_guid(imported.guid).serialize_document()
        artifact = project / f'Library/Artifacts/Texture/{imported.guid}.inxtex'
        assert artifact.is_file()
        old_artifact = artifact.read_bytes()
        sidecar = Path(str(source) + '.meta')
        captured_document = sidecar.read_bytes()
        if change == 'create':
            sidecar.unlink()
        png(source, 3)
        png(peer, 3)
        database.begin_refresh()
        deadline = time.monotonic() + 20
        while str(source).replace('\\', '/') not in database.last_refresh_imported_paths:
            assert time.monotonic() < deadline, 'scan did not capture updated source'
            assert not database.try_commit_refresh(), 'refresh unexpectedly completed'
            time.sleep(.001)
        while not database.last_refresh_importer_task_count:
            assert time.monotonic() < deadline, 'metadata capture did not complete'
            assert not database.try_commit_refresh(), 'refresh unexpectedly completed'
            time.sleep(.001)

        # No more owner-thread progress occurs until the external write is done.
        expected = sidecar.read_bytes() if sidecar.exists() else None
        if change == 'delete':
            sidecar.unlink()
            expected = None
        elif change == 'source':
            png(source, 4)
        elif change != 'none':
            stamp = sidecar.stat() if sidecar.exists() else None
            document = json.loads(captured_document)
            document['metadata']['srgb']['value'] = False
            encoded = (json.dumps(document, indent=4) + '\n').encode('utf-8')
            if change == 'same-stat':
                # JSON whitespace compensates true/false length; neither stat
                # timestamp nor size alone can identify this author revision.
                encoded = json.dumps(document, separators=(',', ':')).encode('utf-8')
                assert len(encoded) <= len(expected)
                encoded += b' ' * (len(expected) - len(encoded))
            if change == 'replace':
                incoming = sidecar.with_name('incoming.meta')
                incoming.write_bytes(encoded)
                os.replace(incoming, sidecar)
            else:
                sidecar.write_bytes(encoded)
            if change == 'same-stat':
                os.utime(sidecar, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                assert sidecar.stat().st_size == stamp.st_size
                assert sidecar.stat().st_mtime_ns == stamp.st_mtime_ns
            expected = encoded

        error = None
        try:
            database.complete_pending_refresh()
        except RuntimeError as exc:
            error = exc
        if change == 'none':
            assert error is None
            assert database.query_generation > generation
            assert artifact.read_bytes() != old_artifact
        else:
            assert (sidecar.read_bytes() if sidecar.exists() else None) == expected
            assert error is not None, 'stale refresh must explicitly reject publication'
            assert database.query_generation == generation
            assert database.get_meta_by_guid(imported.guid).serialize_document() == old_metadata
            assert artifact.read_bytes() == old_artifact
            assert peer_artifact.read_bytes() == peer_bytes
            assert peer_sidecar.read_bytes() == peer_metadata
            assert not (project / 'Library/AssetRefresh.transaction').exists()
    finally:
        engine.exit()


if __name__ == '__main__':
    exercise(Path(sys.argv[1]), sys.argv[2])
