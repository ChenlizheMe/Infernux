from Infernux.engine.player_package_audit import _is_format_marker_group, _is_runtime_license_group


def test_package_local_format_markers_can_repeat():
    assert _is_format_marker_group([
        "Runtime.inxrt::packaging/py.typed", "Parallel.inxmod::numba/py.typed",
    ])


def test_retired_python_compute_aot_markers_are_not_accepted_as_format_markers():
    assert not _is_format_marker_group([
        "Content.inxpkg::Library/Compute/a/one/0/__version__",
        "Content.inxpkg::Library/Compute/a/two/0/__version__",
    ])
    assert not _is_format_marker_group([
        "Content.inxpkg::Library/Compute/a/one/kernel.spv",
        "Content.inxpkg::Library/Compute/a/two/kernel.spv",
    ])
    assert not _is_format_marker_group([
        "Content.inxpkg::Library/Compute/a/one/__version__",
        "Content.inxpkg::Assets/copied.txt",
    ])


def test_runtime_preserves_repeated_dependency_licenses():
    assert _is_runtime_license_group([
        "Game_Data/Runtime.inxrt::numpy/_licenses/numpy-2.5.2/licenses/numpy/ma/LICENSE",
        "Game_Data/Runtime.inxrt::numpy/ma/LICENSE",
    ])
    assert _is_runtime_license_group([
        "Game_Data/Runtime.inxrt::numpy/random/LICENSE.md",
        "Game_Data/Parallel.inxmod::dependency/LICENSE.md",
    ])


def test_runtime_license_exception_does_not_hide_code_or_game_assets():
    for other in (
        "Runtime.inxrt::numpy/copied.dll", "Runtime.inxrt::numpy/LICENSE.py",
        "Content.inxpkg::Assets/LICENSE", "loose/LICENSE",
    ):
        assert not _is_runtime_license_group(["Runtime.inxrt::numpy/ma/LICENSE", other])
    assert not _is_runtime_license_group([])
