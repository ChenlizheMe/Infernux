"""Measure the real worker/owner stages of one large model Apply."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPOSITORY_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from Infernux.core.asset_types import read_mesh_import_settings  # noqa: E402
from Infernux.core.assets import AssetManager  # noqa: E402
from Infernux.engine.engine import Engine  # noqa: E402
from Infernux.lib import LogLevel, RuntimeMode  # noqa: E402


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_obj_fixture(path: Path, *, triangles: int, minimum_bytes: int) -> None:
    """Write substantial deterministic geometry, then pad with valid comments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"o LargeApplyFixture\n")
        vertex = 1
        batch: list[str] = []
        for index in range(triangles):
            x = index % 1000
            z = index // 1000
            batch.extend((
                f"v {x}.0 0.0 {z}.0\n",
                f"v {x}.0 1.0 {z}.0\n",
                f"v {x}.0 0.0 {z + 1}.0\n",
                f"f {vertex} {vertex + 1} {vertex + 2}\n",
            ))
            vertex += 3
            if len(batch) >= 4096:
                stream.write("".join(batch).encode("ascii"))
                batch.clear()
        if batch:
            stream.write("".join(batch).encode("ascii"))
        stream.flush()
        remaining = minimum_bytes - stream.tell()
        padding = b"x" * ((1024 * 1024) - 3)
        while remaining > 0:
            payload = padding[: max(0, min(len(padding), remaining - 3))]
            line = b"# " + payload + b"\n"
            stream.write(line)
            remaining -= len(line)


def measure(*, minimum_mib: int, triangles: int) -> dict[str, Any]:
    minimum_bytes = minimum_mib * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="infernux-large-model-") as temporary:
        project = Path(temporary) / "Project"
        (project / "ProjectSettings").mkdir(parents=True)
        source = project / "Assets" / "Models" / "LargeApply.obj"
        _write_obj_fixture(source, triangles=triangles, minimum_bytes=minimum_bytes)
        source_bytes = source.stat().st_size
        if source_bytes < minimum_bytes:
            raise RuntimeError("large-model fixture did not reach the requested size")

        engine = Engine(engine_log_level=LogLevel.Warn, mode=RuntimeMode.Headless)
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        try:
            imported = AssetManager.import_asset(str(source), database=database)
            if not imported:
                raise RuntimeError(f"initial model import failed: {imported.error}")
            settings = read_mesh_import_settings(str(source))
            started = time.perf_counter()
            database.begin_model_reimport(str(source), settings.to_dict())
            deadline = time.monotonic() + 300.0
            while True:
                result = database.try_commit_model_reimport()
                if result is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("large model Apply exceeded five minutes")
                time.sleep(0.002)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not result or not result.database_committed:
                raise RuntimeError(f"large model Apply failed: {result.error}")
            artifact = (
                project / "Library" / "Artifacts" / "Mesh" / f"{imported.guid}.inxmesh"
            )
            timings = {
                "worker_import_ms": database.last_model_reimport_worker_ms,
                "owner_prepare_staging_ms": database.last_model_reimport_prepare_ms,
                "persistence_transaction_ms": database.last_model_reimport_persistence_ms,
                "live_publication_ms": database.last_model_reimport_live_publication_ms,
            }
            if any(value < 0.0 for value in timings.values()) or sum(timings.values()) <= 0.0:
                raise RuntimeError(f"invalid model Apply timings: {timings}")
            return {
                "schema": "infernux.large_model_reimport_timing",
                "status": "passed",
                "fixture": {
                    "format": ".obj",
                    "triangle_count": triangles,
                    "source_bytes": source_bytes,
                    "artifact_bytes": artifact.stat().st_size,
                },
                "timings": timings,
                "apply_wall_ms": elapsed_ms,
                "atomic_scope": (
                    "One model's metadata and cooked artifacts commit through one "
                    "DocumentTransaction; this is not a global cross-asset atomicity claim."
                ),
            }
        finally:
            engine.exit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimum-source-mib", type=int, default=64)
    parser.add_argument("--triangles", type=int, default=220_000)
    args = parser.parse_args(argv)
    if args.minimum_source_mib < 64:
        parser.error("--minimum-source-mib must be at least 64")
    if args.triangles <= 0:
        parser.error("--triangles must be positive")
    result = measure(
        minimum_mib=args.minimum_source_mib,
        triangles=args.triangles,
    )
    if args.report is not None:
        _write_json_atomic(args.report.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
