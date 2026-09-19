from pathlib import Path


def test_gpu_jit_external_project_inherits_the_root_compilers():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/InfernuxGpuJit.cmake").read_text(encoding="utf-8")

    assert '"-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"' in cmake
    assert '"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"' in cmake
    assert "if(CMAKE_C_COMPILER)" in cmake
    assert "if(CMAKE_CXX_COMPILER)" in cmake


def test_gpu_jit_wheel_build_only_requests_the_shipping_compiler_target():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/InfernuxGpuJit.cmake").read_text(encoding="utf-8")

    build_command = cmake.split("BUILD_COMMAND", 1)[1].split("INSTALL_COMMAND", 1)[0]
    assert "--target _infernux_gpu_compiler" in build_command
    assert "infernux_compiler_contract_tests" not in build_command
    assert "infernux_bit_contract_tests" not in build_command
