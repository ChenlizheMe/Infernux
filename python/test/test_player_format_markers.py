from Infernux.engine.player_package_audit import _is_format_marker_group


def test_package_local_format_markers_can_repeat():
    assert _is_format_marker_group([
        "Content.inxpkg::Library/Compute/a/one/0/__version__",
        "Content.inxpkg::Library/Compute/a/two/0/__version__",
    ])
    assert _is_format_marker_group([
        "Runtime.inxrt::packaging/py.typed", "Parallel.inxmod::numba/py.typed",
    ])


def test_real_payload_duplicates_are_not_format_markers():
    assert not _is_format_marker_group([
        "Content.inxpkg::Library/Compute/a/one/kernel.spv",
        "Content.inxpkg::Library/Compute/a/two/kernel.spv",
    ])
    assert not _is_format_marker_group([
        "Content.inxpkg::Library/Compute/a/one/__version__",
        "Content.inxpkg::Assets/copied.txt",
    ])
