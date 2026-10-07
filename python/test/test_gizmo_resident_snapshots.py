"""Real Vulkan compute/readback and native publication for repeated Gizmo draws."""
import numpy as np
import pytest

import infernux as inx
from infernux.gizmos import Gizmos
from infernux.engine.project_context import using_project_root


@pytest.fixture(params=['scalar', 'vector3'])
def positions(request, engine, monkeypatch, tmp_path):
    host = engine._acquire_compute_host()
    monkeypatch.setattr(inx.compute, '_native_compute_host', lambda: host)
    values = np.asarray([[0, 0, 0], [5, 0, 0], [0, 5, 0]], dtype=np.float32)
    source = inx.buffer(shape=(3,) if request.param == 'vector3' else values.shape,
                        dtype=inx.vector3 if request.param == 'vector3' else np.float32,
                        device='gpu', data=values)
    try:
        Gizmos._begin_frame()
        with using_project_root(tmp_path):
            yield source
    finally:
        engine.clear_component_gizmos()
        Gizmos._begin_frame()
        source.close()
        inx.compute._release_engine_resources()
        host._release_lease()


def snapshots():
    batches = list(Gizmos._resident_draw_batches)
    arrays = []
    for state, _ in batches:
        snapshot = state.vertices.get_data()
        try:
            arrays.append(snapshot.numpy(copy=True))
        finally:
            snapshot.close()
    return batches, arrays


@pytest.mark.parametrize('kind', ['colors', 'matrices', 'radii'])
def test_repeated_resident_draws_preserve_state_and_reach_native(engine, positions, kind):
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.uint32)
    with inx.compute._record_commands():
        Gizmos.color = (1, 0, 0)
        if kind == 'radii':
            Gizmos.draw_wire_spheres(positions, .2, segments=8)
        else:
            Gizmos.draw_lines(positions, edges)
        Gizmos.color = (0, 0, 1)
        if kind == 'matrices':
            Gizmos.color = (1, 0, 0)
            Gizmos.matrix = list(Gizmos._identity_matrix)
            Gizmos.matrix[12] = 3
        if kind == 'radii':
            Gizmos.draw_wire_spheres(positions, .8, segments=8)
        else:
            Gizmos.draw_lines(positions, edges)
    batches, values = snapshots()
    assert len(batches) == 2
    np.testing.assert_allclose(values[0][:, 10:13], np.tile([1, 0, 0], (len(values[0]), 1)))
    np.testing.assert_allclose(values[1][:, 10:13], np.tile([1, 0, 0] if kind == 'matrices' else [0, 0, 1],
                                                         (len(values[1]), 1)))
    if kind == 'radii':
        np.testing.assert_allclose([np.linalg.norm(v[0, :3]) for v in values], [.2, .8], atol=1e-6)
    if kind == 'matrices':
        assert batches[0][1][12] == 0 and batches[1][1][12] == 3
        Gizmos.matrix[12] = 99
        assert batches[1][1][12] == 3
    descriptors = Gizmos._get_resident_data()
    assert len({descriptor[0] for descriptor in descriptors}) == 2
    engine.upload_component_resident_gizmos(descriptors)


@pytest.mark.parametrize('change', ['in_place', 'new_array', 'noncontiguous'])
def test_resident_line_topology_is_captured_at_each_call(engine, positions, change):
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.uint32)
    original = edges.copy()
    with inx.compute._record_commands():
        Gizmos.draw_lines(positions, edges)
        if change == 'in_place':
            edges[:] = [[2, 1], [1, 0]]
            second = edges
        elif change == 'new_array':
            second = np.asarray([[2, 1], [1, 0]], dtype=np.uint32)
        else:
            second = np.asarray([[2, 0], [1, 2]], dtype=np.uint32).T
        expected_second = second.copy()
        Gizmos.draw_lines(positions, second)
        second[:] = 0
    descriptors = Gizmos._get_resident_data()
    np.testing.assert_array_equal(descriptors[0][3], original.ravel())
    np.testing.assert_array_equal(descriptors[1][3], expected_second.ravel())
    assert descriptors[0][0] != descriptors[1][0]
    engine.upload_component_resident_gizmos(descriptors)


@pytest.mark.parametrize('shape', ['lines', 'spheres'])
def test_transient_topologies_reuse_frame_slots_and_retire_with_source(engine, positions, shape):
    identities = None
    retained = []
    for frame in range(12):
        with inx.compute._record_commands():
            Gizmos._begin_frame()
            for draw in range(2):
                if shape == 'lines':
                    Gizmos.draw_lines(positions, np.asarray([[0, draw + 1]], dtype=np.uint32))
                else:
                    Gizmos.draw_wire_spheres(positions, .2 + .1 * draw, segments=8,
                                            center_indices=np.asarray([draw], dtype=np.int32))
        batches, values = snapshots()
        current = [state.identity for state, _ in batches]
        assert len(set(current)) == 2
        if identities is None:
            identities = current
            retained = [state for state, _ in batches]
        else:
            assert current == identities, 'temporary NumPy topology allocated another retained GPU state'
        engine.upload_component_resident_gizmos(Gizmos._get_resident_data())
        engine.clear_component_gizmos()
    positions.close()
    assert all(state.domain.closed and state.vertices.closed for state in retained)


@pytest.mark.parametrize('replace', [False, True])
def test_wire_sphere_selector_is_captured_for_each_draw(engine, positions, replace):
    selectors = np.asarray([0], dtype=np.int32)
    with inx.compute._record_commands():
        Gizmos.draw_wire_spheres(positions, .2, segments=8, center_indices=selectors)
        if replace:
            selectors = np.asarray([1], dtype=np.int32)
        else:
            selectors[:] = 1
        Gizmos.draw_wire_spheres(positions, .8, segments=8, center_indices=selectors)
    _, values = snapshots()
    np.testing.assert_allclose(values[0][0, :3], [0, .2, 0], atol=1e-6)
    np.testing.assert_allclose(values[1][0, :3], [5, .8, 0], atol=1e-6)
    engine.upload_component_resident_gizmos(Gizmos._get_resident_data())


@pytest.mark.parametrize('shape', ['lines', 'spheres'])
def test_resident_source_writes_are_ordered_between_draws(engine, positions, shape):
    edges = np.asarray([[0, 1]], dtype=np.uint32)
    moved = np.asarray([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float32)
    with inx.compute._record_commands():
        if shape == 'lines':
            Gizmos.draw_lines(positions, edges)
        else:
            Gizmos.draw_wire_spheres(positions, .2, segments=8)
        positions.set_data(moved)
        if shape == 'lines':
            Gizmos.draw_lines(positions, edges)
        else:
            Gizmos.draw_wire_spheres(positions, .2, segments=8)
    _, values = snapshots()
    offset = [0, .2, 0] if shape == 'spheres' else [0, 0, 0]
    np.testing.assert_allclose(values[0][0, :3], offset, atol=1e-6)
    np.testing.assert_allclose(values[1][0, :3], moved[0] + offset, atol=1e-6)
    engine.upload_component_resident_gizmos(Gizmos._get_resident_data())


def test_topology_updates_refresh_identity_keep_vertex_allocation(engine, positions):
    identity = vertices = None
    for edges in ([[0, 1]], [[0, 2]], [[0, 1], [1, 2]]):
        Gizmos._begin_frame()
        Gizmos.draw_lines(positions, np.asarray(edges, dtype=np.uint32))
        state, _ = Gizmos._resident_draw_batches[0]
        if vertices is not None:
            assert state.identity != identity, 'native topology cache needs a fresh identity'
            assert state.vertices is vertices, 'changed edges do not require new vertex storage'
        identity, vertices = state.identity, state.vertices
        engine.upload_component_resident_gizmos(Gizmos._get_resident_data())
        Gizmos._begin_frame()
        Gizmos.draw_lines(positions, np.asarray(edges, dtype=np.int64))
        assert Gizmos._resident_draw_batches[0][0].identity == identity
        engine.upload_component_resident_gizmos(Gizmos._get_resident_data())


def test_sphere_layout_changes_close_replaced_slots(engine, positions):
    previous = None
    for segments, selectors in ((8, [0]), (12, [0]), (12, [0, 1]), (8, [1])):
        Gizmos._begin_frame()
        Gizmos.draw_wire_spheres(positions, .2, segments=segments,
                                center_indices=np.asarray(selectors, dtype=np.int32))
        state, _ = Gizmos._resident_draw_batches[0]
        if previous is not None:
            assert previous.domain.closed and previous.vertices.closed
            assert state.identity != previous.identity
        previous = state
        _, values = snapshots()
        assert len(values[0]) == segments * 3 * len(selectors)
        np.testing.assert_allclose(values[0][0, :3], [selectors[0] * 5, .2, 0], atol=1e-6)
        engine.upload_component_resident_gizmos(Gizmos._get_resident_data())
    assert len(positions._gizmo_wire_sphere_pool.states) == 1


def test_sphere_selector_switches_between_explicit_and_all(engine, positions):
    identity = None
    for selectors in (None, np.asarray([2, 0, 1], dtype=np.int32), None):
        Gizmos._begin_frame()
        Gizmos.draw_wire_spheres(positions, .2, segments=8, center_indices=selectors)
        batches, values = snapshots()
        if identity is not None:
            assert batches[0][0].identity == identity
        identity = batches[0][0].identity
        expected = [0, .2, 0] if selectors is None else [0, 5.2, 0]
        np.testing.assert_allclose(values[0][0, :3], expected, atol=1e-6)
        engine.upload_component_resident_gizmos(Gizmos._get_resident_data())
