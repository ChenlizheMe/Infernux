"""ImGui shader generation must preserve unchanged build inputs."""
from pathlib import Path
import os
import subprocess
import tempfile

import pytest

from tests.tool_discovery import find_executable


ROOT = Path(__file__).resolve().parents[2]


def test_shader_generation_survives_reconfigure_and_rebuilds_changed_source():
    fixture_root = ROOT / "out/pytest-cmake"
    fixture_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=fixture_root) as temporary:
        _check_shader_build(Path(temporary))


def _check_shader_build(tmp_path):
    cmake = find_executable("cmake")
    glslang = find_executable("glslangValidator")
    if not glslang and os.environ.get("VULKAN_SDK"):
        candidate = Path(os.environ["VULKAN_SDK"]) / "Bin/glslangValidator.exe"
        if candidate.is_file():
            glslang = str(candidate)
    if not cmake or not glslang:
        pytest.skip("CMake and glslangValidator are required")

    external = (ROOT / "external/CMakeLists.txt").read_text(encoding="utf-8")
    start = external.index("add_custom_command(\n    OUTPUT \"${INFERNUX_IMGUI_FRAGMENT_HEADER}\"")
    command = external[start:external.index("\n)", start) + 2]
    shader = tmp_path / "fragment.frag"
    shader.write_text("#version 450\nlayout(location=0) out vec4 color;\n"
                      "void main() { color = vec4(1.0); }\n", encoding="utf-8")
    (tmp_path / "consumer.cpp").write_text(
        'const unsigned int shader[] = {\n#include "fragment.u32"\n};\n', encoding="utf-8"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\nproject(ShaderGeneration LANGUAGES CXX)\n"
        f'set(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE "{Path(glslang).as_posix()}")\n'
        'set(INFERNUX_IMGUI_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")\n'
        'set(INFERNUX_IMGUI_FRAGMENT_HEADER "${INFERNUX_IMGUI_GENERATED_DIR}/fragment.u32")\n'
        'set(INFERNUX_IMGUI_FRAGMENT_SOURCE "${CMAKE_SOURCE_DIR}/fragment.frag")\n'
        + command
        + '\nadd_library(shader STATIC consumer.cpp "${INFERNUX_IMGUI_FRAGMENT_HEADER}")\n'
        + 'target_include_directories(shader PRIVATE "${INFERNUX_IMGUI_GENERATED_DIR}")\n',
        encoding="utf-8",
    )
    binary = tmp_path / "build"

    def run(*arguments):
        result = subprocess.run([cmake, *arguments], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr

    run("-S", str(tmp_path), "-B", str(binary))
    run("--build", str(binary))
    header = binary / "generated/fragment.u32"
    original = header.read_bytes(), header.stat().st_mtime_ns
    run("-S", str(tmp_path), "-B", str(binary))
    run("--build", str(binary))
    assert (header.read_bytes(), header.stat().st_mtime_ns) == original

    shader.write_text(shader.read_text(encoding="utf-8").replace("vec4(1.0)", "vec4(0.25)"),
                      encoding="utf-8")
    run("--build", str(binary))
    assert header.read_bytes() != original[0]
