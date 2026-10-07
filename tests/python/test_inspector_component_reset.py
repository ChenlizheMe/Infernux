"""Inspector Reset runs on the actual component; history replays its document."""
from types import SimpleNamespace

import pytest

from infernux.components import InxComponent, serialized_field, BoxCollider
from infernux.engine.bootstrap_inspector._wire import (
    _Ctx, _wire_cache_init, _wire_component_list, _wire_clipboard_and_context,
)
from infernux.engine.interaction import EditorInteractionCore, SelectionDomain, SelectionTarget
from infernux.engine.ui import inspector_support
from infernux.engine.undo import UndoManager
from infernux.lib import SceneManager, InspectorComponentInfo, Vector3


@pytest.fixture
def inspector(engine, scene):
    core = EditorInteractionCore()
    history = UndoManager(core.action_journal)
    core.panels.register_selection_authority('inspector', (SelectionDomain.COMPONENT,))
    ctx = _Ctx()
    ctx.ip = SimpleNamespace()
    ctx.engine = engine
    ctx.bs = SimpleNamespace(interaction_core=core)
    ctx.SceneManager = SceneManager
    ctx.InspectorComponentInfo = InspectorComponentInfo
    ctx.InxComponent = InxComponent
    ctx._inspector_support = inspector_support
    ctx._bump_inspector_values = inspector_support.bump_inspector_value_generation
    ctx._record_profile_count = inspector_support.record_inspector_profile_count
    ctx._record_profile_timing = inspector_support.record_inspector_profile_timing
    ctx._t = lambda value: value
    ctx.get_component_icon_id = lambda *_args: 0
    ctx.can_set_component_enabled = ctx.set_component_enabled = None
    _wire_cache_init(ctx)
    _wire_component_list(ctx)
    _wire_clipboard_and_context(ctx)
    yield ctx.bs._inspector_component_actions, history, core
    core.shutdown()


@pytest.mark.parametrize('enabled', [False, True])
def test_reset_has_live_owner_siblings_identity_and_declared_defaults(inspector, scene, enabled):
    actions, history, _core = inspector
    calls = []
    class OwnerReset(InxComponent):
        value: int = serialized_field(default=2)
        untouched: int = serialized_field(default=4)
        def reset(self):
            owner = self.game_object
            calls.append((int(owner.id), int(self.component_id), self.value))
            assert self in owner.get_py_components()
            assert owner.get_component('BoxCollider') is not None
            self.value += len(owner.name)
        def on_before_serialize(self):
            # The default bag must never receive owner-dependent callbacks.
            assert self.game_object is not None
    owner = scene.create_game_object('PuzzleDoor')
    owner.add_component(BoxCollider)
    component = owner.add_component(OwnerReset)
    component.enabled = enabled
    component.value, component.untouched = 87, 91
    calls.clear()
    before_count = len(owner.get_py_components())
    identity = component.component_id
    assert actions.reset((owner.id, identity, False))
    assert calls == [(int(owner.id), int(identity), 2)]
    assert (component.value, component.untouched) == (12, 4)
    assert component.enabled is enabled
    assert len(owner.get_py_components()) == before_count
    history.undo()
    assert (component.value, component.untouched) == (87, 91)
    history.redo()
    assert (component.value, component.untouched) == (12, 4)
    assert len(calls) == 1 and component.component_id == identity


@pytest.mark.parametrize('failure', ['raise', 'invalid'])
def test_failed_reset_restores_fields_and_does_not_publish_history(inspector, scene, failure):
    actions, history, _core = inspector
    class FailedReset(InxComponent):
        value: int = serialized_field(default=2)
        values: list[int] = serialized_field(default=[1, 2])
        def reset(self):
            if getattr(self, '_fail', False):
                self.value = 19
                if failure == 'raise':
                    raise RuntimeError('expected Reset failure')
                self.values.append('not an integer')
    owner = scene.create_game_object('Failure')
    component = owner.add_component(FailedReset)
    component.value = 87
    component.values = [7, 8]
    component._fail = True
    before = len(tuple(history.action_journal.applied_entries()))
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        actions.reset((owner.id, component.component_id, False))
    assert component.value == 87
    assert component.values == [7, 8]
    assert len(tuple(history.action_journal.applied_entries())) == before


@pytest.mark.parametrize('fail_second', [False, True])
def test_reset_batch_is_one_field_document_action(inspector, scene, fail_second):
    actions, history, core = inspector
    class BatchReset(InxComponent):
        value: int = serialized_field(default=2)
        def reset(self):
            self.value = len(self.game_object.name)
            if getattr(self, '_fail', False):
                raise RuntimeError('expected second Reset failure')
    components = [scene.create_game_object(name).add_component(BatchReset)
                  for name in ('First', 'Second')]
    for component in components:
        component.value = 87
    components[1]._fail = fail_second
    core.selection.replace([SelectionTarget.component(c.game_object.id, c.component_id, sub_kind='script')
                            for c in components], owner_id='inspector', record_history=False)
    before = len(tuple(history.action_journal.applied_entries()))
    if fail_second:
        with pytest.raises(RuntimeError, match='reset callback failed'):
            actions.reset()
        assert [c.value for c in components] == [87, 87]
        assert len(tuple(history.action_journal.applied_entries())) == before
    else:
        assert actions.reset()
        assert [c.value for c in components] == [5, 6]
        assert len(tuple(history.action_journal.applied_entries())) == before + 1
        history.undo()
        assert [c.value for c in components] == [87, 87]
        history.redo()
        assert [c.value for c in components] == [5, 6]


def test_native_reset_keeps_identity_and_undo(inspector, scene):
    actions, history, _core = inspector
    owner = scene.create_game_object('NativeReset')
    component = owner.add_component(BoxCollider)
    component.size = Vector3(4, 5, 6)
    identity = component.component_id
    assert actions.reset((owner.id, identity, True))
    assert tuple(component.size) == (1, 1, 1)
    history.undo()
    assert tuple(component.size) == (4, 5, 6)
    assert component.component_id == identity


def test_unchanged_reset_does_not_add_history(inspector, scene):
    actions, history, _core = inspector
    class PlainReset(InxComponent):
        value: int = serialized_field(default=2)
    owner = scene.create_game_object('Unchanged')
    component = owner.add_component(PlainReset)
    before = len(tuple(history.action_journal.applied_entries()))
    assert not actions.reset((owner.id, component.component_id, False))
    assert len(tuple(history.action_journal.applied_entries())) == before
