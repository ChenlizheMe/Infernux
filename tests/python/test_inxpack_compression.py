from __future__ import annotations

import importlib
import os
import threading
import time

import pytest

from infernux.engine.player_package_native import read_entry, write_pack_isolated


def _native_inxpack():
    module = importlib.import_module("infernux.lib._Infernux")
    if not hasattr(module, "_inxpack_write"):
        pytest.skip("the native InxPack binding is not installed")
    return module


def test_inxpack_compression_profile_roundtrip_and_determinism(tmp_path):
    native = _native_inxpack()
    source = tmp_path / "payload.bin"
    source.write_bytes((b"infernux-release-payload\0" * 4096) + bytes(range(256)))
    files = [("Runtime/payload.bin", str(source))]

    development_path = tmp_path / "development.inxrt"
    development_again_path = tmp_path / "development-again.inxrt"
    release_path = tmp_path / "release.inxrt"
    development_manifest = native._inxpack_write(files, str(development_path))
    development_again_manifest = native._inxpack_write(files, str(development_again_path))
    release_manifest = native._inxpack_write(files, str(release_path), profile="release")
    explicit_manifest = native._inxpack_write(files, str(tmp_path / "explicit.inxrt"), compression_level=6)

    assert development_path.read_bytes() == development_again_path.read_bytes()
    assert development_manifest["archive_sha256"] == development_again_manifest["archive_sha256"]
    assert release_manifest["archive_sha256"] == explicit_manifest["archive_sha256"]
    assert native._inxpack_read_entry(str(release_path), "Runtime/payload.bin") == source.read_bytes()


def test_isolated_inxpack_worker_roundtrip(tmp_path):
    source = tmp_path / "isolated-source.bin"
    source.write_bytes(b"isolated-player-content" * 4096)
    destination = tmp_path / "isolated.inxpkg"
    polls: list[int] = []

    manifest = write_pack_isolated(
        [("Library/isolated-source.bin", source)],
        destination,
        profile="development",
        cancel_event=threading.Event(),
        on_wait=lambda: polls.append(1),
    )

    assert polls
    assert manifest["file_count"] == 1
    assert read_entry(destination, "Library/isolated-source.bin") == source.read_bytes()


@pytest.mark.parametrize("compression_level", [-1, 0, 23])
def test_inxpack_rejects_invalid_explicit_compression_level(tmp_path, compression_level):
    native = _native_inxpack()
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    with pytest.raises((ValueError, RuntimeError)):
        native._inxpack_write(
            [("Runtime/payload.bin", str(source))],
            str(tmp_path / "invalid.inxrt"),
            compression_level=compression_level,
        )


def test_inxpack_reads_and_extraction_allow_other_python_threads(tmp_path):
    native = _native_inxpack()
    source = tmp_path / "large.bin"
    block = os.urandom(4 * 1024 * 1024)
    with source.open("wb") as stream:
        for _ in range(16):
            stream.write(block)
    pack = str(tmp_path / "large.inxpkg")
    native._inxpack_write([("large.bin", str(source))], pack, compression_level=1)
    ticks: list[float] = []
    started, stopped = threading.Event(), threading.Event()

    def heartbeat():
        started.set()
        while not stopped.wait(.001):
            ticks.append(time.monotonic())

    worker = threading.Thread(target=heartbeat)
    worker.start()
    started.wait()
    try:
        # Reading this one-entry manifest is sub-millisecond; it cannot be used
        # as a scheduler timing test. Exercise the actual large payload IO.
        for name, operation in (
            ("read", lambda: native._inxpack_read_entry(pack, "large.bin")),
            ("extract", lambda: native._inxpack_extract(pack, str(tmp_path / "extracted"))),
        ):
            before = time.monotonic()
            operation()
            after = time.monotonic()
            assert any(before < tick < after for tick in ticks), f"{name} held the GIL for {after - before:.3f}s"
    finally:
        stopped.set()
        worker.join()
