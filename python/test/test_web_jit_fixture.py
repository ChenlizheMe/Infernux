from __future__ import annotations

import ast
import json
from pathlib import Path
import py_compile

import pytest

from Infernux._jit_kernels import build_interpreted_cpu_source
from Infernux.engine.player_package_native import write_pack
from scripts.acceptance.verify_web_jit_fixture import (
    GAME_DATA,
    SCRIPT_BYTECODE,
    verify_web_jit_fixture,
)


def test_web_fixture_public_jit_cooks_to_an_executable_python_function():
    source = (
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/multiplatform_player/Assets/Scripts/Bootstrap.py"
    ).read_text(encoding="utf-8")
    cooked = build_interpreted_cpu_source(source)
    tree = ast.parse(cooked)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "cpu_jit_probe"
    )
    assert function.decorator_list == []
    assert "jit.warmup" not in cooked
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fixture", "exec"), namespace)
    assert namespace["cpu_jit_probe"](7) == 84


def _sealed_fixture(
    tmp_path: Path, source: str, *, content_files: tuple[str, ...] = ()
) -> tuple[Path, Path]:
    script = tmp_path / "Bootstrap.py"
    bytecode = tmp_path / "Bootstrap.pyc"
    script.write_text(source, encoding="utf-8")
    py_compile.compile(str(script), cfile=str(bytecode), doraise=True)
    content = tmp_path / "Content.inxpkg"
    entries = [(SCRIPT_BYTECODE, bytecode)]
    for index, path in enumerate(content_files):
        payload = tmp_path / f"content-{index}.bin"
        payload.write_bytes(b"compiler payload")
        entries.append((path, payload))
    write_pack(entries, content)
    output = tmp_path / "web"
    output.mkdir()
    write_pack(
        ((f"{GAME_DATA}/Content.inxpkg", content),),
        output / "infernux-player.0123456789abcdef01234567.inxpkg",
    )
    report = tmp_path / "build-report.json"
    report.write_text(
        json.dumps({
            "status": "passed",
            "target": "web-wasm32",
            "manifest": {"python_runtime": {"packages": [
                {"name": "numpy", "imports": ["numpy"]}
            ]}},
        }),
        encoding="utf-8",
    )
    return output, report


def test_web_jit_fixture_verifier_accepts_sealed_plain_python(tmp_path):
    output, report = _sealed_fixture(
        tmp_path,
        "def cpu_jit_probe(count):\n    return count * 12\n",
    )
    result = verify_web_jit_fixture(output, report)
    assert result["cpu_function"] == "cpu_jit_probe"
    assert result["compiler_payloads"] == 0


def test_web_jit_fixture_verifier_rejects_live_warmup(tmp_path):
    output, report = _sealed_fixture(
        tmp_path,
        "import infernux as inx\n"
        "def cpu_jit_probe(count):\n    return count * 12\n"
        "inx.jit.warmup(cpu_jit_probe, 7)\n",
    )
    with pytest.raises(RuntimeError, match="retained a CPU compiler or warmup call"):
        verify_web_jit_fixture(output, report)


@pytest.mark.parametrize(
    "payload_path",
    (
        "Assets/numba/__init__.pyc",
        "Packages/llvmlite/binding/__init__.pyc",
        "Library/LLVM/compiler.bin",
        "Library/Parallel.inxmod",
    ),
)
def test_web_jit_fixture_verifier_rejects_nested_compiler_payload(
    tmp_path, payload_path
):
    output, report = _sealed_fixture(
        tmp_path,
        "def cpu_jit_probe(count):\n    return count * 12\n",
        content_files=(payload_path,),
    )
    with pytest.raises(RuntimeError, match="content contains a CPU compiler payload"):
        verify_web_jit_fixture(output, report)
