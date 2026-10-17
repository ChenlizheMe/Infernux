"""Pure-logic tests for the editor UI layer (no ImGui frame required).

Covers: theme color math + C++ SSOT override, viewport math, window manager
state machine, panel state persistence, scene-view math helpers, UI-editor
geometry, and selection manager behavior.
"""

from __future__ import annotations


import pytest as pytest


class _WindowManagerPanelInteractions:
    """Explicit interaction boundary for WindowManager-only tests."""

    def bind_view(self, _view_id, _type_id, _instance):
        pass

    def unbind_view(self, _view_id):
        return True

    def records_focus_history(self, **_identity):
        return True

    def is_document_backed(self, **_identity):
        return False


def _window_manager(engine, panel_interactions=None):
    from infernux.engine.ui.window_manager import WindowManager

    registrar = getattr(engine, "register_gui", None)
    if registrar is None:
        registrar = lambda _window_id, _instance: None
    return WindowManager(
        engine,
        panel_interactions or _WindowManagerPanelInteractions(),
        registrar,
    )


class _FakeCtx:
    """Minimal stand-in exposing the mouse-position API ViewportInfo uses."""

    def __init__(self, x: float, y: float):
        self._x, self._y = x, y

    def get_mouse_pos_x(self) -> float:
        return self._x

    def get_mouse_pos_y(self) -> float:
        return self._y


class TestWindowManager:
    @staticmethod
    def _layout_reset_fixture():
        from infernux.engine.interaction import (
            DocumentCapability,
            DocumentKind,
            EditorInteractionCore,
            PanelInteractionDescriptor,
        )
        from infernux.engine.ui.closable_panel import ClosablePanel
        from infernux.engine.ui.editor_panel import EditorPanel

        class Engine:
            def __init__(self):
                self.registered = {}
                self.reset_count = 0
                self.focused = []

            def register_gui(self, window_id, instance):
                self.registered[window_id] = instance

            def unregister_gui(self, window_id):
                self.registered.pop(window_id, None)

            def reset_imgui_layout(self):
                self.reset_count += 1

            def select_docked_window(self, window_id, *, allow_during_modal=False):
                self.focused.append((window_id, bool(allow_during_modal)))

        class Controller:
            def __init__(self):
                self.discard_calls = 0

            def discard(self, *, document_id):
                assert document_id == "graph-document"
                self.discard_calls += 1
                return True

        core = EditorInteractionCore()
        engine = Engine()
        manager = _window_manager(engine, core.panels)
        for type_id in ("scene_view", "graph"):
            core.panels.register_type(type_id, PanelInteractionDescriptor())

        scene = EditorPanel("Scene", "scene_view")
        manager.register_existing_window("scene_view", scene, "scene_view")
        manager.register_window_type(
            "graph",
            ClosablePanel,
            "Graph",
            factory=lambda: ClosablePanel("Graph", "graph"),
        )
        graph = manager.open_window("graph")
        manager.process_pending_actions()

        controller = Controller()
        document = core.documents.create(
            DocumentKind.GENERIC,
            "Graph",
            document_id="graph-document",
            revision=1,
            saved_revision=0,
            capabilities=DocumentCapability.DISCARD,
            controller=controller,
        )
        graph.bind_document(document.document_id)
        return core, engine, manager, scene, graph, document, controller


class TestUICanvasCollectionCache:
    class _Root:
        def __init__(self, canvas):
            self.id = id(self)
            self._components = [] if canvas is None else [canvas]

        def get_py_components(self):
            return list(self._components)

        @staticmethod
        def get_children():
            return []

    class _Scene:
        name = "SameName"
        structure_version = 7

        def __init__(self, canvas):
            self._root = TestUICanvasCollectionCache._Root(canvas)
            self.world_id = id(self)
            self.temporal_discontinuity_revision = 0
            self.root_queries = 0
            if canvas is not None:
                canvas._set_game_object(self._root)
                canvas._native_scene = self
                canvas._try_get_game_object = lambda: self._root

        def get_root_objects(self):
            self.root_queries += 1
            return [self._root]
