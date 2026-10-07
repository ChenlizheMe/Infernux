"""Prepared publication must preserve the decisions made by lifecycle callbacks."""
from pathlib import Path

import pytest

from infernux.components.script_loader import load_and_create_component
from infernux.engine.component_restore import (
    deserialize_game_object_document_transactionally,
    deserialize_scene_document_transactionally,
)
from infernux.lib import AssetRegistry, SceneManager


SOURCE = '''from infernux.components import InxComponent, serialized_field
class PreparedDecision(InxComponent):
    decision: str = serialized_field(default="keep")
    def awake(self):
        self.events = ["awake"]
        if self.decision == "disable":
            self.enabled = False
        elif self.decision == "enable":
            self.enabled = True
        elif self.decision == "owner_disable":
            self.game_object.active = False
        elif self.decision == "error":
            raise RuntimeError("expected prepared Awake failure")
    def on_enable(self):
        self.events.append("enable")
        if self.decision == "enable_disable":
            self.enabled = False
    def on_disable(self):
        self.events.append("disable")
    def start(self):
        self.events.append("start")
    def update(self, dt):
        self.events.append("update")
'''


@pytest.fixture
def script(tmp_path):
    database = AssetRegistry.instance().get_asset_database()
    path = Path(database.assets_root) / (tmp_path.name + '.py')
    assert not path.exists()
    path.write_text(SOURCE, encoding='utf-8')
    guid = database.import_asset(str(path)).guid
    assert guid
    try:
        yield path, guid, database
    finally:
        database.delete_asset(str(path))
        path.unlink()
        Path(str(path) + '.meta').unlink(missing_ok=True)


@pytest.mark.parametrize('drive', ['add', 'object', 'scene'])
@pytest.mark.parametrize('decision,authored', [
    ('keep', True), ('keep', False), ('disable', True), ('enable', False),
    ('enable_disable', True), ('owner_disable', True), ('error', True),
])
def test_publication_keeps_lifecycle_decisions(scene, script, drive, decision, authored):
    path, guid, database = script
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    owner = scene.create_game_object('PreparedOwner')
    instance = load_and_create_component(str(path), asset_database=database,
                                         script_guid=guid, type_name='PreparedDecision')
    assert instance is not None
    instance.enabled = authored
    instance.decision = decision if drive == 'add' else 'keep'
    component = owner.add_py_component(instance)
    component_id = component.component_id
    owner_id = owner.id
    if drive != 'add':
        component.decision = decision
        if drive == 'object':
            assert deserialize_game_object_document_transactionally(
                owner, owner.serialize_document(), asset_database=database)
        else:
            assert deserialize_scene_document_transactionally(
                scene, scene.serialize_document(), asset_database=database)
        owner = scene.find_by_id(owner_id)
        component = owner.get_py_components()[0]

    expected_enabled = (decision == 'enable' or authored) and decision not in (
        'disable', 'enable_disable', 'error')
    assert component.component_id == component_id
    assert component.enabled is expected_enabled
    assert component._cpp_component.enabled is expected_enabled
    assert owner.active is (decision != 'owner_disable')
    expected_events = ['awake']
    if decision == 'enable_disable':
        expected_events += ['enable', 'disable']
    elif expected_enabled and owner.active:
        expected_events += ['enable']
    assert component.events == expected_events

    if drive == 'scene':
        # A full Scene transaction publishes an unstarted world. The runtime
        # scene service explicitly starts that world before its first frame.
        manager._start_scene_for_play(scene)
    for _ in range(2):
        manager.step(1.0 / 60.0)
    if expected_enabled and owner.active:
        expected_events += ['start', 'update', 'update']
    assert component.events == expected_events

    # Explicit author action can enable a dormant component later. Awake is
    # still once-only, and Start belongs to its first effective activation.
    component.decision = 'keep'
    component.enabled = True
    owner.active = True
    for _ in range(2):
        manager.step(1.0 / 60.0)
    assert component.events.count('awake') == 1
    assert component.events.count('start') == 1
    assert component.events.count('update') == (4 if expected_enabled and decision != 'owner_disable' else 2)
