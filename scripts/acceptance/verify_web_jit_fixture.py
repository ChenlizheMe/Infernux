#!/usr/bin/env python3
"""Inspect the sealed Web fixture for a cooked public CPU JIT function."""

from __future__ import annotations

import argparse
import importlib.util
import json
import marshal
from pathlib import Path
import sys
import tempfile
from types import CodeType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from Infernux.engine.player_package_native import read_entry, read_manifest  # noqa: E402


GAME_DATA = "InfernuxPlatformFixture_Data"
SCRIPT_BYTECODE = "Assets/Scripts/Bootstrap.pyc"


def _code_objects(code: CodeType):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _code_objects(value)


def verify_web_jit_fixture(output: Path, build_report: Path) -> dict[str, object]:
    report = json.loads(build_report.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("target") != "web-wasm32":
        raise RuntimeError("Web fixture build report is not a passed Web target")
    runtime = report["manifest"]["python_runtime"]
    packages = runtime["packages"]
    forbidden_imports = {"numba", "llvmlite", "llvm"}
    runtime_imports = {
        name.casefold() for package in packages
        for name in (package["name"], *package["imports"])
    }
    if runtime_imports & forbidden_imports:
        raise RuntimeError("Web Python runtime includes a CPU compiler package")

    outer_packages = sorted(output.glob("infernux-player.*.inxpkg"))
    if len(outer_packages) != 1:
        raise RuntimeError("Web fixture must publish exactly one Player package")
    outer = outer_packages[0]
    outer_paths = {record["path"] for record in read_manifest(outer)["files"]}
    forbidden = sorted(
        path for path in outer_paths
        if any(part.casefold() in {"numba", "llvmlite", "llvm", "parallel.inxmod"}
               for part in Path(path).parts)
    )
    if forbidden:
        raise RuntimeError(f"Web fixture contains a CPU compiler payload: {forbidden}")

    content_path = f"{GAME_DATA}/Content.inxpkg"
    if content_path not in outer_paths:
        raise RuntimeError(f"Web fixture has no sealed content package: {content_path}")
    with tempfile.TemporaryDirectory(prefix="infernux-web-jit-") as temporary:
        content = Path(temporary) / "Content.inxpkg"
        content.write_bytes(read_entry(outer, content_path))
        content_paths = {record["path"] for record in read_manifest(content)["files"]}
        if SCRIPT_BYTECODE not in content_paths:
            raise RuntimeError(f"Web fixture has no cooked script: {SCRIPT_BYTECODE}")
        if "Assets/Scripts/Bootstrap.py" in content_paths:
            raise RuntimeError("Web fixture shipped the authoring Python source")
        bytecode = read_entry(content, SCRIPT_BYTECODE)

    if bytecode[:4] != importlib.util.MAGIC_NUMBER:
        raise RuntimeError("Web fixture script bytecode uses a different Python ABI")
    code = marshal.loads(bytecode[16:])
    if not isinstance(code, CodeType):
        raise RuntimeError("Web fixture script is not Python code")
    code_objects = tuple(_code_objects(code))
    if not any(item.co_name == "cpu_jit_probe" for item in code_objects):
        raise RuntimeError("Web fixture lost its public CPU JIT probe")
    if any("jit" in item.co_names or "warmup" in item.co_names for item in code_objects):
        raise RuntimeError("Web fixture retained a CPU compiler or warmup call")
    return {
        "player_package": outer.name,
        "content_package": content_path,
        "script": SCRIPT_BYTECODE,
        "cpu_function": "cpu_jit_probe",
        "compiler_payloads": 0,
        "runtime_packages": [package["name"] for package in packages],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-report", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_web_jit_fixture(args.output, args.build_report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
