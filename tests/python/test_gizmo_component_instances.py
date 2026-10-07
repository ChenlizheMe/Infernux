"""Real component/physics geometry at the collector's CPU upload boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from infernux.gizmos import Gizmos
from infernux.gizmos.collector import GizmosCollector
from infernux.lib import Physics, Vector3


class GeometryRecorder:
    def __init__(self):
        self.world = np.empty((0, 3), dtype=np.float32)
        self.icons = []

    def upload_component_gizmos(self, vertices, count, indices, descriptors, draws):
        vertices = np.asarray(vertices).reshape(count, 6)[:, :3]
        result = []
        for row in np.asarray(descriptors).reshape(draws, 18):
            start, size = map(int, row[:2])
            points = vertices[np.asarray(indices)[start:start + size]]
            matrix = row[2:].reshape(4, 4).T
            result.append(np.c_[points, np.ones(len(points))] @ matrix.T)
        self.world = np.concatenate(result)[:, :3]

    def clear_component_cpu_gizmos(self):
        self.world = np.empty((0, 3), dtype=np.float32)

    def upload_component_resident_gizmos(self, descriptors):
        assert not descriptors

    def upload_component_gizmo_icons(self, *args):
        self.icons = list(Gizmos._icon_entries)

    def clear_component_gizmo_icons(self):
        self.icons = []

    def clear_component_gizmos(self):
        self.clear_component_cpu_gizmos()
        self.clear_component_gizmo_icons()


def setup_collector(selected):
    recorder = GeometryRecorder()
    collector = GizmosCollector()
    frontend = SimpleNamespace(get_native_engine=lambda: recorder,
                               get_selected_object_id=lambda: selected[0])
    return recorder, collector, lambda: collector.collect_and_upload(frontend)


def assert_centers(recorder, centers):
    if not centers:
        assert len(recorder.world) == 0
        return
    assert len(recorder.world) > 0
    for center in centers:
        assert np.any(np.abs(recorder.world[:, 0] - center) < 1.1), (centers, recorder.world)
    assert np.all(np.min(np.abs(recorder.world[:, 0, None] - np.asarray(centers)), axis=1) < 1.1)


@pytest.mark.parametrize('kind', ['BoxCollider', 'SphereCollider', 'CapsuleCollider'])
@pytest.mark.parametrize('state', ['both', 'first_disabled', 'second_disabled', 'both_disabled',
                                  'first_removed', 'inactive_parent', 'unselected'])
def test_every_enabled_collider_contributes_geometry(scene, kind, state):
    parent = scene.create_game_object('Compound parent')
    owner = scene.create_game_object('Compound collider')
    owner.set_parent(parent, False)
    first, second = owner.add_component(kind), owner.add_component(kind)
    first.center, second.center = Vector3(0, 0, 0), Vector3(4, 0, 0)
    assert first.component_id != second.component_id
    if state in ('first_disabled', 'both_disabled'):
        first.enabled = False
    if state in ('second_disabled', 'both_disabled'):
        second.enabled = False
    if state == 'first_removed':
        assert owner.remove_component(first)
    if state == 'inactive_parent':
        parent.active = False
    recorder, collector, collect = setup_collector([0 if state == 'unselected' else parent.id])
    try:
        collect()
        expected = {'both': [0, 4], 'first_disabled': [4], 'second_disabled': [0],
                    'both_disabled': [], 'first_removed': [4], 'inactive_parent': [], 'unselected': []}[state]
        assert_centers(recorder, expected)
        if state in ('both', 'first_disabled', 'first_removed'):
            hit = Physics.raycast(Vector3(4, 0, -3), Vector3(0, 0, 1))
            assert hit is not None and hit.game_object == owner
    finally:
        collector.retire_uploaded(recorder)


@pytest.mark.parametrize('kind', ['BoxCollider', 'SphereCollider', 'CapsuleCollider'])
def test_collider_instance_changes_and_reparent_refresh_same_collector(scene, kind):
    parent = scene.create_game_object('Original parent')
    owner = scene.create_game_object('Mutable compound')
    owner.set_parent(parent, False)
    first, second = owner.add_component(kind), owner.add_component(kind)
    second.center = Vector3(4, 0, 0)
    selected = [parent.id]
    recorder, collector, collect = setup_collector(selected)
    try:
        collect()
        assert_centers(recorder, [0, 4])
        first.enabled = False
        collect()
        assert_centers(recorder, [4])
        assert owner.remove_component(first)
        replacement = owner.add_component(kind)
        replacement.center = Vector3(-4, 0, 0)
        collect()
        assert_centers(recorder, [-4, 4])
        destination = scene.create_game_object('Destination')
        destination.transform.position = Vector3(20, 0, 0)
        owner.set_parent(destination, False)
        selected[0] = destination.id
        collect()
        assert_centers(recorder, [16, 24])
        destination.active = False
        collect()
        assert_centers(recorder, [])
        destination.active = True
        collect()
        assert_centers(recorder, [16, 24])
    finally:
        collector.retire_uploaded(recorder)


def test_multiple_light_instances_keep_one_icon_per_owner_and_type(scene):
    owner = scene.create_game_object('Two lights')
    first, second = owner.add_component('Light'), owner.add_component('Light')
    second.enabled = False
    recorder, collector, collect = setup_collector([owner.id])
    try:
        collect()
        single_count = len(recorder.world)
        assert single_count > 0
        second.enabled = True
        collect()
        assert len(recorder.world) == single_count * 2
        assert sum(entry[1] == owner.id for entry in recorder.icons) == 1
        first.enabled = False
        collect()
        assert len(recorder.world) == single_count
        assert sum(entry[1] == owner.id for entry in recorder.icons) == 1
    finally:
        collector.retire_uploaded(recorder)
