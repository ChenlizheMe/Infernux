"""UI Editor projections and gesture retirement on real native scene objects."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


def run_case(tmp_path, *args):
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), str(tmp_path), *args],
        env=dict(os.environ), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'UI_EDITOR_LIFECYCLE_OK' in result.stdout


@pytest.mark.parametrize('world', ['active', 'additive'])
@pytest.mark.parametrize('kind', ['UIImage', 'UIText'])
@pytest.mark.parametrize('action', ['before', 'after_empty', 'reselect', 'remove', 'readd', 'unrelated', 'stable'])
def test_selected_component_membership(tmp_path, world, kind, action):
    run_case(tmp_path, 'selection', kind, action, world)


@pytest.mark.parametrize('kind', ['drag', 'resize', 'rotate'])
@pytest.mark.parametrize('action', ['finish', 'hidden', 'disable', 'close_lifecycle', 'cancel'])
def test_gesture_retirement_and_undo(tmp_path, kind, action):
    run_case(tmp_path, 'gesture', kind, action)


@pytest.mark.parametrize('kind', ['pan', 'canvas', 'zoom'])
@pytest.mark.parametrize('action', ['hidden', 'disable'])
def test_view_gesture_retirement(tmp_path, kind, action):
    run_case(tmp_path, 'view', kind, action)


def exercise(project, group, kind, action, world='active'):
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui.ui_editor_panel import UIEditorPanel
    from infernux.engine.ui.core_panel_interactions import ui_editor_panel_interaction
    from infernux.engine.undo import UndoManager
    from infernux.lib import LogLevel, RuntimeMode, SceneManager, Vector3
    from infernux import ui

    (project / 'Assets').mkdir()
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    panel = None
    try:
        engine.init_headless(str(project))
        panel = UIEditorPanel()
        panel.set_engine(engine)
        core = EditorInteractionCore.instance()
        core.panels.register_type('ui_editor', ui_editor_panel_interaction(core.scene_objects))
        core.panels.bind_view('ui_editor', 'ui_editor', panel)
        scene_manager = SceneManager.instance()
        scene = scene_manager.get_active_scene()
        if world == 'additive':
            active_scene = scene
            scene = scene_manager.create_scene('Other Room')
            scene_manager.set_active_scene(active_scene)
        canvas = scene.create_game_object('Canvas')
        canvas.add_component(ui.UICanvas)
        owner = scene.create_game_object('Selected')
        owner.set_parent(canvas)
        assert core.selection.select_scene_object(owner.id, owner_id='ui_editor', record_history=False)
        assert core.focus.activate_panel('ui_editor', view_id='ui_editor', record_history=False)
        if group == 'selection':
            if action in ('after_empty', 'reselect'):
                assert panel._selected_element_comp is None
            component = owner.add_component(getattr(ui, kind))
            if action == 'reselect':
                core.selection.select_scene_object(canvas.id, owner_id='ui_editor', record_history=False)
                core.selection.select_scene_object(owner.id, owner_id='ui_editor', record_history=False)
            assert panel._selected_element_comp is component
            if action in ('remove', 'readd'):
                assert owner.remove_component(component)
                assert panel._selected_element_comp is None
                if action == 'readd':
                    replacement = owner.add_component(getattr(ui, kind))
                    assert replacement.component_id != component.component_id
                    assert panel._selected_element_comp is replacement
            if action == 'unrelated':
                other = scene.create_game_object('Other')
                other.add_component(ui.UIImage)
                assert panel._selected_element_comp is component
                assert panel._element_object_id(component) == owner.id
            if action == 'stable':
                # Count the real enumeration boundary, without replacing the
                # selected object, its components or the projection itself.
                owner_type = type(owner)
                get_components = owner_type.get_py_components
                calls = []
                def counted_components(current):
                    calls.append(current.id)
                    return get_components(current)
                owner_type.get_py_components = counted_components
                try:
                    for index in range(50):
                        component.width = 100 + index
                        assert panel._selected_element_comp is component
                    assert calls == [], 'stable property edits must not scan component membership'
                finally:
                    owner_type.get_py_components = get_components
            return

        if group == 'view':
            before = panel._capture_view_state()
            manager = UndoManager.instance()
            old_entries = len(manager.action_journal.applied_entries())
            key = panel._begin_continuous_view_edit(kind, 'Test view change')
            if kind == 'pan':
                panel._is_panning = True
                panel._pan_x += 30
            elif kind == 'canvas':
                panel._dragging_canvas = True
                panel._drag_canvas_id = canvas.id
                panel._canvas_panel_positions[canvas.id] = [30, 45]
            else:
                panel._zoom *= .8
            panel._update_continuous_view_edit(key)
            after = panel._capture_view_state()
            if action == 'hidden':
                panel._on_not_visible(None)
            else:
                panel.on_disable()
            assert core.continuous_edits.get(key) is None
            assert not panel._is_panning and not panel._dragging_canvas and not panel._drag_canvas_id
            assert len(manager.action_journal.applied_entries()) == old_entries + 1
            manager.undo()
            assert panel._capture_view_state() == before
            manager.redo()
            assert panel._capture_view_state() == after
            panel.on_disable()
            assert len(manager.action_journal.applied_entries()) == old_entries + 1
            return

        component = owner.add_component(ui.UIImage)
        assert panel._selected_element_comp is component
        before = panel._element_manipulation_snapshot(kind, component)
        manager = UndoManager.instance()
        old_entries = len(manager.action_journal.applied_entries())
        assert panel._begin_element_manipulation(kind, component)
        key = panel._active_element_edit_key
        assert core.continuous_edits.get(key) is not None
        assert core.focus.snapshot.capture_owner_id == 'ui_editor.manipulation'
        setattr(panel, {'drag': '_dragging', 'resize': '_resizing', 'rotate': '_rotating'}[kind], True)
        def mutate():
            if kind == 'drag':
                owner.transform.local_position = Vector3(17, 23, 0)
            elif kind == 'rotate':
                owner.transform.local_euler_angles = Vector3(0, 0, 37)
            else:
                component.width += 31
                component.height += 17
        assert panel._mutate_element_manipulation(mutate)
        after = panel._element_manipulation_snapshot(kind, component)
        assert after != before
        if action == 'hidden':
            panel._on_not_visible(None)
        elif action == 'disable':
            panel.on_disable()
        elif action == 'close_lifecycle':
            panel._enable_called = True
            panel._finalize_close_lifecycle()
            assert panel._enable_called is False
        else:
            assert panel._finish_element_manipulation(commit=action == 'finish')
        assert not panel._active_element_edit_key
        assert core.continuous_edits.get(key) is None
        assert core.focus.snapshot.capture_owner_id == ''
        if action == 'cancel':
            assert panel._element_manipulation_snapshot(kind, component) == before
            assert len(manager.action_journal.applied_entries()) == old_entries
        else:
            assert len(manager.action_journal.applied_entries()) == old_entries + 1
            manager.undo()
            assert panel._element_manipulation_snapshot(kind, component) == before
            assert core.focus.snapshot.capture_owner_id == ''
            manager.redo()
            assert panel._element_manipulation_snapshot(kind, component) == after
            assert core.focus.snapshot.capture_owner_id == ''
        if action in ('hidden', 'disable', 'close_lifecycle'):
            assert not any((panel._dragging, panel._resizing, panel._rotating))
            panel.on_disable()
            assert panel._begin_element_manipulation(kind, component), 'a reopened panel can acquire input again'
            assert panel._finish_element_manipulation(commit=False)
        # Retiring an already idle UI Editor cannot release another owner.
        core.focus.set_capture_owner('other-panel')
        panel.on_disable()
        assert core.focus.snapshot.capture_owner_id == 'other-panel'
        core.focus.set_capture_owner('')
    finally:
        if panel is not None:
            panel._finish_element_manipulation(commit=False)
        engine.exit()


if __name__ == '__main__':
    exercise(Path(sys.argv[1]), *sys.argv[2:])
    print('UI_EDITOR_LIFECYCLE_OK')
