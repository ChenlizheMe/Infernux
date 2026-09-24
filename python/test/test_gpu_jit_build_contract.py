from pathlib import Path


def test_source_checkout_gpu_vendor_follows_the_loaded_native_directory(tmp_path, monkeypatch):
    from Infernux import lib as engine_lib
    from Infernux._compiler import taichi as compiler_loader

    native_dir = tmp_path / "Release"
    vendor = tmp_path / "gpu-jit-wheel/Infernux/_compiler/taichi/_vendor/taichi"
    native_dir.mkdir()
    vendor.mkdir(parents=True)
    monkeypatch.delenv("INFERNUX_GPU_JIT_VENDOR_DIR", raising=False)
    monkeypatch.delenv("INFERNUX_NATIVE_MODULE_DIR", raising=False)
    monkeypatch.setattr(engine_lib, "native_dir", str(native_dir))

    assert compiler_loader._vendor_dir() == vendor


def test_gpu_jit_external_project_inherits_the_root_compilers():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/InfernuxGpuJit.cmake").read_text(encoding="utf-8")

    assert '"-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"' in cmake
    assert '"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"' in cmake
    assert "if(CMAKE_C_COMPILER)" in cmake
    assert "if(CMAKE_CXX_COMPILER)" in cmake


def test_gpu_jit_external_build_installs_only_the_shipping_compiler_target():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake/InfernuxGpuJit.cmake").read_text(encoding="utf-8")

    build_command = cmake.split("BUILD_COMMAND", 1)[1].split("INSTALL_COMMAND", 1)[0]
    assert "--target ${_infernux_gpu_jit_build_targets}" in build_command
    assert "set(_infernux_gpu_jit_build_targets _infernux_gpu_compiler)" in cmake
    assert "if(INFERNUX_BUILD_TESTS)" in cmake
    assert "infernux_compiler_contract_tests" in cmake
    assert "infernux_bit_contract_tests" in cmake
    install_command = cmake.split("INSTALL_COMMAND", 1)[1]
    assert "--component infernux_compiler" in install_command


def test_gpu_jit_native_boundary_has_one_private_module_export_surface():
    root = Path(__file__).resolve().parents[2]
    fork = root / "external/taichi_for_infernux"
    policy = (fork / "cmake/InfernuxJitPolicy.cmake").read_text(encoding="utf-8")
    platform = (fork / "taichi/common/platform_macros.h").read_text(encoding="utf-8")
    core = (fork / "cmake/TaichiCore.cmake").read_text(encoding="utf-8")
    binding = (fork / "taichi/python/export.cpp").read_text(encoding="utf-8")

    assert "add_compile_definitions(TI_INFERNUX_PRIVATE_COMPILER=1)" in policy
    assert "defined(TI_INFERNUX_PRIVATE_COMPILER)" in platform
    assert "#define TI_DLL_EXPORT\n" in platform
    assert "file(GLOB TAICHI_CORE_SOURCE" not in core
    assert "file(GLOB TAICHI_PYBIND_SOURCE" not in core
    assert "gfx_runtime" not in core
    assert "aot" not in core.casefold()
    assert "PYBIND11_MODULE(_infernux_gpu_compiler, m)" in binding
    assert "InterfaceHolder" not in binding
