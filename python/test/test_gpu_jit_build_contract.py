from pathlib import Path


def test_gpu_jit_external_project_inherits_the_root_compilers():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/InfernuxGpuJit.cmake").read_text(encoding="utf-8")

    assert '"-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"' in cmake
    assert '"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"' in cmake
    assert "if(CMAKE_C_COMPILER)" in cmake
    assert "if(CMAKE_CXX_COMPILER)" in cmake
