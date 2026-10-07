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
    owners = [scene.create_game_object('Batch A'), scene.create_game_object('Batch B')]
    transforms = [owner.transform for owner in owners]
    target = transforms if mode == 'list' else create_batch_handle(transforms, mode=mode)
    return owners, transforms, target


def expected_data(prop):
    if prop in ('rotation', 'local_rotation'):
        return np.asarray([[0, 0, 0, 1], [0, .6, 0, .8]], dtype=np.float32)
    return np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)


@pytest.mark.parametrize('mode', ['list', 'strict', 'compact'])
@pytest.mark.parametrize('prop', ['local_position', 'local_rotation'])
@pytest.mark.parametrize('layout', ['valid', 'wide', 'narrow', 'zero_columns', 'rank1', 'rank3',
                                   'short_rows', 'strided', 'reversed', 'readonly', 'extra_rows'])
def test_transform_batch_shape_is_atomic(scene, mode, prop, layout):
    _, transforms, target = targets_in(scene, mode)
    expected = expected_data(prop)
    values = expected.copy()
    if layout == 'wide':
        values = np.column_stack((expected, np.full(2, 99, dtype=np.float32)))
    elif layout == 'narrow':
        values = expected[:, :-1].copy()
    elif layout == 'zero_columns':
        values = np.empty((2, 0), dtype=np.float32)
    elif layout == 'rank1':
        values = expected.reshape(-1)
    elif layout == 'rank3':
        values = expected.reshape(2, 1, -1)
    elif layout == 'short_rows':
        values = expected[:1]
    elif layout == 'strided':
        storage = np.full((2, expected.shape[1] * 2), -99, dtype=np.float32)
        storage[:, ::2] = expected
        values = storage[:, ::2]
        assert not values.flags.c_contiguous
    elif layout == 'reversed':
        values = values[::-1]
        expected = expected[::-1]
    elif layout == 'readonly':
        values.flags.writeable = False
    elif layout == 'extra_rows':
        values = np.vstack((expected, expected[:1]))
    original_values = values.copy()
    before = batch_read(transforms, prop).copy()
    if layout in ('wide', 'narrow', 'zero_columns', 'rank1', 'rank3', 'short_rows'):
        with pytest.raises(ValueError):
            batch_write(target, values, prop)
        np.testing.assert_array_equal(batch_read(transforms, prop), before)
        # A rejected call must leave the handle reusable.
        batch_write(target, expected, prop)
    else:
        mask = batch_write(target, values, prop)
        if mode == 'compact':
            np.testing.assert_array_equal(mask, [True, True])
    np.testing.assert_allclose(batch_read(transforms, prop), expected, atol=1e-5)
    np.testing.assert_array_equal(values, original_values)


@pytest.mark.parametrize('mode', ['list', 'strict', 'compact'])
@pytest.mark.parametrize('prop', ['local_position', 'position', 'local_scale', 'local_euler_angles',
                                 'euler_angles', 'local_rotation', 'rotation'])
def test_all_transform_batch_fields_preserve_rows(scene, mode, prop):
    _, transforms, target = targets_in(scene, mode)
    values = expected_data(prop)
    batch_write(target, values, prop)
    np.testing.assert_allclose(batch_read(transforms, prop), values, atol=1e-5)


@pytest.mark.parametrize('mode', ['strict', 'compact'])
@pytest.mark.parametrize('prop', ['local_position', 'local_rotation'])
@pytest.mark.parametrize('removed', [1, 2])
def test_stale_batch_members_preserve_compaction_indices(scene, mode, prop, removed):
    owners, transforms, target = targets_in(scene, mode)
    expected = expected_data(prop)
    before = batch_read(transforms, prop).copy()
    for owner in owners[:removed]:
        scene.destroy_game_object(owner)
    scene.process_pending_destroys()
    if mode == 'strict':
        with pytest.raises(RuntimeError, match='stale transform'):
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


@pytest.mark.parametrize('mode', ['strict', 'compact'])
@pytest.mark.parametrize('prop', ['local_position', 'local_rotation'])
def test_empty_handle_has_explicit_element_shape(mode, prop):
    target = create_batch_handle([], mode=mode)
    width = expected_data(prop).shape[1]
    result = batch_read(target, prop)
    values = result[0] if mode == 'compact' else result
    assert values.shape == (0, width)
    batch_write(target, np.empty((0, width), dtype=np.float32), prop)
    with pytest.raises(ValueError):
        batch_write(target, np.empty((0, width + 1), dtype=np.float32), prop)


@pytest.mark.parametrize('entry', ['strict', 'compact', 'read', 'write'])
@pytest.mark.parametrize('invalid', ['null', 'wrong_type'])
def test_native_batch_rejects_missing_transform(tmp_path, entry, invalid):
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), str(tmp_path), entry, invalid],
        env=dict(os.environ), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TRANSFORM_BATCH_REJECTION_OK' in result.stdout


def exercise_invalid(project, entry, invalid):
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    (project / 'Assets').mkdir()
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(lib.LogLevel.Warn, lib.RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        scene = lib.SceneManager.instance().get_active_scene()
        owner = scene.create_game_object('Survivor')
        transform = owner.transform
        transform.local_position = lib.Vector3(7, 8, 9)
        targets = [transform, None if invalid == 'null' else 'not a Transform']
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            if entry in ('strict', 'compact'):
                create_batch_handle(targets, mode=entry)
            elif entry == 'read':
                lib._transform_batch_read(targets, 'local_position')
            else:
                lib._transform_batch_write(targets, expected_data('local_position'), 'local_position')
        np.testing.assert_array_equal(batch_read([transform], 'local_position'), [[7, 8, 9]])
        good = create_batch_handle([transform])
        batch_write(good, np.asarray([[3, 4, 5]], dtype=np.float32), 'local_position')
        np.testing.assert_array_equal(batch_read(good, 'local_position'), [[3, 4, 5]])
    finally:
        engine.exit()


if __name__ == '__main__':
    exercise_invalid(Path(sys.argv[1]), *sys.argv[2:])
    print('TRANSFORM_BATCH_REJECTION_OK')
