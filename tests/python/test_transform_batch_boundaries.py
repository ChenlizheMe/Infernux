"""Native Transform batches reject invalid input before touching scene storage."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from infernux import lib
from infernux.batch import batch_read, batch_write, create_batch_handle


def targets_in(scene, mode):
    owners = [scene.create_game_object("Batch A"), scene.create_game_object("Batch B")]
    transforms = [owner.transform for owner in owners]
    target = transforms if mode == "list" else create_batch_handle(transforms, mode=mode)
    return owners, transforms, target


def expected_data(prop):
    if prop in ("rotation", "local_rotation"):
        return np.asarray([[0, 0, 0, 1], [0, 0.6, 0, 0.8]], dtype=np.float32)
    return np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)


@pytest.mark.parametrize(
    "mode,prop,layout",
    [
        ("list", "local_position", "valid"),
        ("strict", "local_rotation", "wide"),
        ("compact", "local_rotation", "zero_columns"),
        ("list", "local_position", "strided"),
        ("compact", "local_rotation", "readonly"),
    ],
)
def test_transform_batch_shape_is_atomic(scene, mode, prop, layout):
    _, transforms, target = targets_in(scene, mode)
    expected = expected_data(prop)
    values = expected.copy()
    if layout == "wide":
        values = np.column_stack((expected, np.full(2, 99, dtype=np.float32)))
    elif layout == "narrow":
        values = expected[:, :-1].copy()
    elif layout == "zero_columns":
        values = np.empty((2, 0), dtype=np.float32)
    elif layout == "rank1":
        values = expected.reshape(-1)
    elif layout == "rank3":
        values = expected.reshape(2, 1, -1)
    elif layout == "short_rows":
        values = expected[:1]
    elif layout == "strided":
        storage = np.full((2, expected.shape[1] * 2), -99, dtype=np.float32)
        storage[:, ::2] = expected
        values = storage[:, ::2]
        assert not values.flags.c_contiguous
    elif layout == "reversed":
        values = values[::-1]
        expected = expected[::-1]
    elif layout == "readonly":
        values.flags.writeable = False
    elif layout == "extra_rows":
        values = np.vstack((expected, expected[:1]))
    original_values = values.copy()
    before = batch_read(transforms, prop).copy()
    if layout in ("wide", "narrow", "zero_columns", "rank1", "rank3", "short_rows"):
        with pytest.raises(ValueError):
            batch_write(target, values, prop)
        np.testing.assert_array_equal(batch_read(transforms, prop), before)
        # A rejected call must leave the handle reusable.
        batch_write(target, expected, prop)
    else:
        mask = batch_write(target, values, prop)
        if mode == "compact":
            np.testing.assert_array_equal(mask, [True, True])
    np.testing.assert_allclose(batch_read(transforms, prop), expected, atol=1e-5)
    np.testing.assert_array_equal(values, original_values)


@pytest.mark.parametrize(
    "mode,prop",
    [
        ("list", "local_position"),
        ("strict", "position"),
        ("compact", "local_scale"),
        ("list", "local_euler_angles"),
        ("strict", "euler_angles"),
        ("compact", "local_rotation"),
        ("strict", "rotation"),
    ],
)
def test_all_transform_batch_fields_preserve_rows(scene, mode, prop):
    _, transforms, target = targets_in(scene, mode)
    values = expected_data(prop)
    batch_write(target, values, prop)
    np.testing.assert_allclose(batch_read(transforms, prop), values, atol=1e-5)


@pytest.mark.parametrize(
    "mode,prop,removed",
    [
        ("strict", "local_position", 1),
        ("strict", "local_rotation", 2),
        ("compact", "local_rotation", 1),
        ("compact", "local_position", 2),
    ],
)
def test_stale_batch_members_preserve_compaction_indices(scene, mode, prop, removed):
    owners, transforms, target = targets_in(scene, mode)
    expected = expected_data(prop)
    before = batch_read(transforms, prop).copy()
    for owner in owners[:removed]:
        scene.destroy_game_object(owner)
    scene.process_pending_destroys()
    if mode == "strict":
        with pytest.raises(RuntimeError, match="stale transform"):
            batch_write(target, expected, prop)
        if removed == 1:
            np.testing.assert_allclose(batch_read([transforms[1]], prop), before[1:])
    else:
        mask = batch_write(target, expected, prop)
        np.testing.assert_array_equal(mask, [False, removed == 1])
        gathered, read_mask = batch_read(target, prop)
        np.testing.assert_array_equal(read_mask, mask)
        np.testing.assert_allclose(gathered, expected[removed:], atol=1e-5)
        # Even an all-stale batch must validate the authored input stride.
        with pytest.raises(ValueError):
            batch_write(target, np.ones((2, expected.shape[1] + 1), dtype=np.float32), prop)


@pytest.mark.parametrize("mode,prop", [("strict", "local_rotation"), ("compact", "local_position")])
def test_empty_handle_has_explicit_element_shape(mode, prop):
    target = create_batch_handle([], mode=mode)
    width = expected_data(prop).shape[1]
    result = batch_read(target, prop)
    values = result[0] if mode == "compact" else result
    assert values.shape == (0, width)
    batch_write(target, np.empty((0, width), dtype=np.float32), prop)
    with pytest.raises(ValueError):
        batch_write(target, np.empty((0, width + 1), dtype=np.float32), prop)


@pytest.mark.parametrize("mode", ["strict", "compact"])
@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("prop", ["local_position", "local_rotation"])
def test_scene_transfer_invalidates_batch_world_identity_without_other_allocations(scene, mode, warm_cache, prop):
    manager = lib.SceneManager.instance()
    destination = manager.create_scene("Batch move destination")
    moved = scene.create_game_object("Moved root")
    child = scene.create_game_object("Moved descendant")
    child.set_parent(moved, False)
    first = scene.create_game_object("First survivor")
    second = scene.create_game_object("Second survivor")
    transforms = [moved.transform, first.transform, child.transform, second.transform]
    handle = create_batch_handle(transforms, mode=mode)
    before = batch_read(transforms, prop).copy()
    if warm_cache:
        batch_read(handle, prop)

    manager.move_game_object_to_scene(moved, destination)
    values = np.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] if prop == "local_position" else
        [[0, 0, 0, 1], [0, 0.6, 0, 0.8], [0.6, 0, 0, 0.8], [0, 0, 0.6, 0.8]],
        dtype=np.float32,
    )
    expected_mask = np.asarray([False, True, False, True])
    if mode == "strict":
        with pytest.raises(RuntimeError, match="stale transform"):
            batch_read(handle, prop)
        with pytest.raises(RuntimeError, match="stale transform"):
            batch_write(handle, values, prop)
        np.testing.assert_array_equal(batch_read(transforms, prop), before)
    else:
        gathered, mask = batch_read(handle, prop)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(gathered, before[expected_mask])
        np.testing.assert_array_equal(batch_write(handle, values, prop), expected_mask)
        gathered, mask = batch_read(handle, prop)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_allclose(gathered, values[expected_mask], atol=1e-5)
        # Both the moved root and its moved child retain their original values;
        # compact rows still address the original input indices 1 and 3.
        np.testing.assert_array_equal(batch_read(transforms, prop)[~expected_mask], before[~expected_mask])

    # An unrelated allocation must not change the outcome of world validation.
    destination.create_game_object("Unrelated")
    if mode == "strict":
        with pytest.raises(RuntimeError, match="stale transform"):
            batch_read(handle, prop)
    else:
        np.testing.assert_array_equal(batch_read(handle, prop)[1], expected_mask)
        manager.move_game_object_to_scene(first, destination)
        manager.move_game_object_to_scene(second, destination)
        gathered, mask = batch_read(handle, prop)
        assert gathered.shape == (0, values.shape[1])
        np.testing.assert_array_equal(mask, False)
        np.testing.assert_array_equal(batch_write(handle, values, prop), False)
        with pytest.raises(ValueError):
            batch_write(handle, values[:, :-1], prop)
        manager.move_game_object_to_scene(first, scene)
        manager.move_game_object_to_scene(second, scene)

    # Returning to the captured world revives the same generational identities.
    manager.move_game_object_to_scene(moved, scene)
    mask = batch_write(handle, values, prop)
    if mode == "compact":
        np.testing.assert_array_equal(mask, True)
    np.testing.assert_allclose(batch_read(transforms, prop), values, atol=1e-5)


def hierarchy_targets(scene):
    parent = scene.create_game_object("Batch parent")
    spacer = scene.create_game_object("Untargeted ancestor")
    spacer.set_parent(parent, False)
    spacer.transform.local_position = lib.Vector3(1, 2, 3)
    spacer.transform.local_euler_angles = lib.Vector3(13, 24, 8)
    child = scene.create_game_object("Batch child")
    child.set_parent(spacer, False)
    leaf = scene.create_game_object("Batch grandchild")
    leaf.set_parent(child, False)
    for owner in (parent, child, leaf):
        owner.transform.euler_angles = lib.Vector3(5, 10, 15)
    return [parent.transform, child.transform, leaf.transform]


@pytest.mark.parametrize("mode", ["list", "strict", "compact"])
@pytest.mark.parametrize("prop", ["position", "rotation", "euler_angles"])
@pytest.mark.parametrize("during_frame", [False, True])
@pytest.mark.parametrize("layout", ["extra_rows", "readonly_strided"])
def test_hierarchy_world_batches_publish_final_values_and_keep_input_unchanged(
    engine, scene, mode, prop, during_frame, layout,
):
    transforms = hierarchy_targets(scene)
    indices = [2, 0, 1, 0, 2]
    targets = [transforms[index] for index in indices]
    handle = targets if mode == "list" else create_batch_handle(targets, mode=mode)
    desired = np.asarray({
        "position": [[3, 8, -2], [-5, 7, 13], [19, -3, 11]],
        "rotation": [[0, 0.6, 0, 0.8], [0.6, 0, 0, 0.8], [0, 0, 0.6, 0.8]],
        "euler_angles": [[120, 330, 15], [25, 58, 29], [31, 320, 74]],
    }[prop], dtype=np.float32)
    values = desired[indices].copy()
    values[:2] = [0, 0, 0, 1] if prop == "rotation" else [9, 27, 35]
    if layout == "extra_rows":
        values = np.vstack((values, values[:1]))
    else:
        storage = np.full((len(values), values.shape[1] * 2), -999, dtype=np.float32)
        storage[:, ::2] = values
        values = storage[:, ::2]
        values.flags.writeable = False
    original = values.copy()
    observed = []

    def publish(_delta=0.0):
        write_mask = batch_write(handle, values, prop)
        observed.append((write_mask, batch_read(handle, prop)))

    if during_frame:
        from infernux.runtime_services import register_update

        manager = lib.SceneManager.instance()
        registration = register_update(publish)
        try:
            manager.play()
            engine.tick(0.0)
        finally:
            registration.close()
            manager.stop()
    else:
        publish()
    assert len(observed) == 1
    write_mask, immediate = observed[0]
    if mode == "compact":
        np.testing.assert_array_equal(write_mask, True)
        immediate, mask = immediate
        np.testing.assert_array_equal(mask, True)
    np.testing.assert_allclose(immediate, desired[indices], atol=1e-4)
    np.testing.assert_allclose(batch_read(transforms, prop), desired, atol=1e-4)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("mode,prop", [("list", "position"), ("strict", "rotation"), ("compact", "euler_angles")])
@pytest.mark.parametrize("layout", ["wide", "rank1", "short_rows"])
def test_hierarchy_world_batch_shape_rejection_leaves_every_target_unchanged(scene, mode, prop, layout):
    transforms = hierarchy_targets(scene)
    targets = [transforms[2], transforms[0], transforms[1]]
    handle = targets if mode == "list" else create_batch_handle(targets, mode=mode)
    before = batch_read(transforms, prop).copy()
    width = 4 if prop == "rotation" else 3
    values = np.ones((3, width), dtype=np.float32)
    if layout == "wide":
        values = np.column_stack((values, values[:, :1]))
    elif layout == "rank1":
        values = values.reshape(-1)
    else:
        values = values[:2]
    original = values.copy()
    with pytest.raises(ValueError):
        batch_write(handle, values, prop)
    np.testing.assert_array_equal(batch_read(transforms, prop), before)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("prop", ["local_position", "local_scale", "local_rotation", "local_euler_angles"])
def test_hierarchical_local_batch_values_keep_local_assignment_semantics(scene, prop):
    transforms = hierarchy_targets(scene)
    targets = [transforms[2], transforms[0], transforms[1], transforms[0]]
    values = np.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] if prop != "local_rotation" else
        [[0, 0, 0, 1], [0, 0.6, 0, 0.8], [0.6, 0, 0, 0.8], [0, 0, 0.6, 0.8]], dtype=np.float32,
    )
    batch_write(targets, values, prop)
    np.testing.assert_allclose(batch_read(transforms, prop), values[[3, 2, 0]], atol=1e-4)


@pytest.mark.parametrize(
    "entry,invalid", [("strict", "null"), ("compact", "wrong_type"), ("read", "null"), ("write", "wrong_type")]
)
def test_native_batch_rejects_missing_transform(tmp_path, entry, invalid):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), entry, invalid],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRANSFORM_BATCH_REJECTION_OK" in result.stdout


def exercise_invalid(project, entry, invalid):
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x8003)
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(lib.LogLevel.Warn, lib.RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        scene = lib.SceneManager.instance().get_active_scene()
        owner = scene.create_game_object("Survivor")
        transform = owner.transform
        transform.local_position = lib.Vector3(7, 8, 9)
        targets = [transform, None if invalid == "null" else "not a Transform"]
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            if entry in ("strict", "compact"):
                create_batch_handle(targets, mode=entry)
            elif entry == "read":
                lib._transform_batch_read(targets, "local_position")
            else:
                lib._transform_batch_write(targets, expected_data("local_position"), "local_position")
        np.testing.assert_array_equal(batch_read([transform], "local_position"), [[7, 8, 9]])
        good = create_batch_handle([transform])
        batch_write(good, np.asarray([[3, 4, 5]], dtype=np.float32), "local_position")
        np.testing.assert_array_equal(batch_read(good, "local_position"), [[3, 4, 5]])
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise_invalid(Path(sys.argv[1]), *sys.argv[2:])
    print("TRANSFORM_BATCH_REJECTION_OK")
