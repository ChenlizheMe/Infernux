from __future__ import annotations
import json
import os
import struct
from dataclasses import replace
from types import SimpleNamespace
import pytest
from infernux.engine.i18n import t
from infernux.engine.interaction import DocumentRegistry
from infernux.engine.ui.graph_document_authoring import (
    GraphDocumentAuthoringModel,
    ParticleEmitterGraphAuthoringModel,
    particle_stage_definition_filter,
)
from infernux.graph import GraphDocument, GraphNodeRecord
from infernux.graph.types import CoordinateSpace, TypeRef, ValueType
from infernux.particle.asset import (
    EmitterSettings,
    ParticleEmitterAsset,
    ParticleEventField,
    ParticleEventFlow,
    ParticleEventType,
    ParticleGraphAsset,
    ParticleParameter,
    particle_attribute_cache_id,
)
from infernux.particle.nodes import (
    PARTICLE_EVENT_ACTIVE_TYPE_ID,
    PARTICLE_EVENT_TRIGGER_TYPE_ID,
    particle_event_payload_port_id,
    particle_graph_node_definitions,
)


@pytest.fixture(autouse=True)
def _isolate_particle_graph_panel_dirty_tracking(monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.engine.undo import UndoManager

    class _AssetDatabase:
        def __init__(self):
            self._paths: dict[str, str] = {}
            self._guids: dict[str, str] = {}

        def register(self, path: str, guid: str = "") -> str:
            normalized = os.path.abspath(str(path))
            identity = guid or self._guids.get(os.path.normcase(normalized)) or f"{len(self._paths) + 1:032x}"
            self._paths[identity] = normalized
            self._guids[os.path.normcase(normalized)] = identity
            return identity

        def get_guid_from_path(self, path: str) -> str:
            normalized = os.path.abspath(str(path))
            key = os.path.normcase(normalized)
            if key not in self._guids and os.path.isfile(normalized):
                self.register(normalized)
            return self._guids.get(key, "")

        def get_path_from_guid(self, guid: str) -> str:
            return self._paths.get(str(guid), "")

        def contains_path(self, path: str) -> bool:
            return bool(self.get_guid_from_path(path))

    database = _AssetDatabase()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(
        AssetManager,
        "reimport_asset",
        classmethod(lambda _cls, path, **_kwargs: bool(database.register(path))),
    )
    monkeypatch.setattr(
        AssetManager,
        "import_asset",
        classmethod(lambda _cls, path, **_kwargs: bool(database.register(path))),
    )

    previous_manager = UndoManager.instance()
    UndoManager()
    DocumentRegistry.instance().close_view("particle_graph_editor")
    try:
        yield database
    finally:
        DocumentRegistry.instance().close_view("particle_graph_editor")
        UndoManager._instance = previous_manager


def _register_asset_reference(path, guid: str) -> None:
    from infernux.core.assets import AssetManager

    AssetManager.require_asset_database().register(str(path), guid)


def _stage_model(document):
    return GraphDocumentAuthoringModel(
        document,
        definition_filter=particle_stage_definition_filter(document.domain),
    )


def _finish_particle_graph_save(panel, path) -> None:
    for pending in tuple(panel._authoring_document_controller._pending_writes.values()):
        pending.io_ticket.wait()
    panel._authoring_document_controller.poll_pending_writes()


def _save_particle_graph(panel, path) -> bool:
    registry = DocumentRegistry.instance()
    document = registry.require(panel.document_id)
    target = os.path.abspath(str(path))
    current = os.path.abspath(document.resource_path) if document.resource_path else ""
    ticket = registry.begin_save(
        document.document_id,
        save_as=not current or os.path.normcase(target) != os.path.normcase(current),
    )
    accepted = panel._save_to(str(path), ticket_id=ticket.ticket_id)
    if accepted:
        _finish_particle_graph_save(panel, path)
    return accepted


def _particle_panel_with_command_core():
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui.particle_graph_editor_panel import (
        ParticleGraphEditorPanel,
    )
    from infernux.engine.undo import UndoManager

    core = EditorInteractionCore()
    manager = UndoManager(core.action_journal)
    panel = ParticleGraphEditorPanel()
    core.panels.register_type(panel.panel_type_id, panel.PANEL_INTERACTION)
    core.panels.bind_view(panel.window_id, panel.panel_type_id, panel)
    panel._graph_selection.bind(core.selection)
    core.focus.activate_panel(
        panel.panel_type_id,
        view_id=panel.window_id,
        document_id=panel.document_id,
        child_context_id=panel.current_child_context_id(),
        record_history=False,
    )
    return panel, core, manager


def _particle_panel_with_history():
    from infernux.engine.interaction import (
        DocumentRegistry,
        EditorContextSnapshot,
        SelectionService,
    )
    from infernux.engine.ui.particle_graph_editor_panel import ParticleGraphEditorPanel
    from infernux.engine.undo import UndoManager

    DocumentRegistry()
    selection = SelectionService()
    manager = UndoManager()
    manager.set_context_hooks(
        lambda: EditorContextSnapshot(selection=selection.snapshot),
        lambda context, phase: (
            selection.apply_snapshot(
                context.selection,
                reason=phase,
                record_history=False,
            ),
            True,
        )[1],
    )
    panel = ParticleGraphEditorPanel()
    panel._graph_selection.bind(selection)
    return panel, manager


def test_inline_idle_render_preserves_values_and_defaults_without_copying(engine):
    import copy
    import sys

    from infernux import lib
    from infernux.core.node_graph import GraphNode, NodeInlineFieldDef, NodeTypeDef
    from infernux.engine.ui.node_graph_view import NodeGraphView, _NodeLayout

    fields = [
        NodeInlineFieldDef("signed", "Signed", "i32", -12),
        NodeInlineFieldDef("unsigned", "Unsigned", "u32", 0xFFFFFFFF),
        NodeInlineFieldDef("scalar", "Scalar", "f32", 1.25),
        NodeInlineFieldDef("vector", "Vector", "vec3", [1.0, 2.0, 3.0]),
        NodeInlineFieldDef("color", "Color", "color", [0.2, 0.4, 0.6, 1.0]),
        NodeInlineFieldDef("asset", "Asset", "asset_ref", {"guid": "", "path_hint": ""}, asset_type="Texture"),
        NodeInlineFieldDef("curve", "Curve", "curve", {"keys": [[0.0, 1.0]]}),
        NodeInlineFieldDef("gradient", "Gradient", "gradient", {"keys": [[0.0, [1.0, 0.5, 0.2, 1.0]]]}),
    ]
    node = GraphNode(data={field.id: copy.deepcopy(field.default) for field in fields})
    layout = _NodeLayout(node, NodeTypeDef("inline", "Inline", inline_fields=fields), sx=4.0, w=50.0)
    default_node = GraphNode()
    default_layout = _NodeLayout(default_node, layout.typedef, sx=4.0, w=50.0)
    defaults = copy.deepcopy([field.default for field in fields])
    original_values = dict(node.data)
    original_data = copy.deepcopy(node.data)
    view = NodeGraphView()
    view._semantic_capture_active = False
    edits = []
    view.on_node_data_changed = lambda *change: edits.append(change)
    copies = []
    renders = []
    errors = []

    def count_copies(frame, event, _argument):
        if (event == "call" and frame.f_code is copy.deepcopy.__code__
                and frame.f_back.f_globals.get("__name__") == "infernux.engine.ui.node_graph_view"):
            copies.append(frame.f_back.f_code.co_name)

    class GuiOnlyPipeline(lib.RenderPipelineCallback):
        def render(self, context, camera):
            pass

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            visible = ctx.begin_window("Inline idle###inline_idle", True, 0)
            try:
                if visible:
                    previous_profile = sys.getprofile()
                    sys.setprofile(count_copies)
                    try:
                        for field in fields:
                            view._draw_inline_field(ctx, layout, field, 30.0, detached=True)
                        # Missing fields display their declaration without materializing
                        # a document edit or acquiring ownership of the shared default.
                        ctx.push_id_str("default")
                        try:
                            for field in fields:
                                view._draw_inline_field(ctx, default_layout, field, 30.0, detached=True)
                        finally:
                            ctx.pop_id()
                    finally:
                        sys.setprofile(previous_profile)
                    renders.append(True)
            except Exception as error:
                errors.append(error)
            finally:
                ctx.dummy(0.0, 0.0)
                ctx.end_window()

    probe = Probe()
    engine.set_render_pipeline(GuiOnlyPipeline())
    engine.register_gui_renderable("test.inline_idle", probe)
    try:
        for _ in range(3):
            engine.tick(1.0 / 60.0)
    finally:
        engine.unregister_gui_renderable("test.inline_idle")
        engine.set_render_pipeline(None)

    assert not errors, errors
    assert renders
    assert copies == []
    assert edits == []
    assert node.data == original_data
    assert default_node.data == {}
    assert all(node.data[key] is value for key, value in original_values.items())
    assert [field.default for field in fields] == defaults


def test_inline_edit_snapshots_keep_nested_values_independent_for_undo():
    from infernux.core.node_graph import GraphNode
    from infernux.engine.ui.node_graph_view import NodeGraphView
    from infernux.engine.undo import LambdaCommand, UndoManager

    old_value = {"keys": [[0.0, [1.0, 0.5, 0.2, 1.0]]]}
    new_value = {"keys": [[0.0, [0.1, 0.2, 0.3, 1.0]]]}
    node = GraphNode(data={"gradient": old_value})
    view = NodeGraphView()
    manager = UndoManager()

    def commit(_uid, field_id, before, after):
        manager.execute(LambdaCommand(
            "Edit inline gradient",
            undo_fn=lambda: node.data.__setitem__(field_id, before),
            redo_fn=lambda: node.data.__setitem__(field_id, after),
        ))

    view.on_node_data_changed = commit
    view._commit_inline_value(node, "gradient", new_value)
    old_value["keys"][0][1][0] = 9.0
    new_value["keys"][0][1][0] = 8.0

    assert node.data["gradient"]["keys"][0][1] == [0.1, 0.2, 0.3, 1.0]
    manager.undo()
    assert node.data["gradient"]["keys"][0][1] == [1.0, 0.5, 0.2, 1.0]
    manager.redo()
    assert node.data["gradient"]["keys"][0][1] == [0.1, 0.2, 0.3, 1.0]


def test_rejected_inline_edit_does_not_mutate_document_or_shared_default():
    from infernux.core.node_graph import GraphNode, NodeInlineFieldDef
    from infernux.engine.ui.node_graph_view import NodeGraphView

    default = {"keys": [[0.0, [1.0, 0.5, 0.2, 1.0]]]}
    field = NodeInlineFieldDef("gradient", "Gradient", "gradient", default)
    node = GraphNode()
    view = NodeGraphView()

    def reject(_uid, _field_id, before, after):
        assert before is None
        after["keys"][0][1][0] = 7.0
        raise ValueError("edit rejected")

    view.on_node_data_changed = reject
    with pytest.raises(ValueError, match="edit rejected"):
        view._commit_inline_value(node, field.id, field.default)

    assert node.data == {}
    assert field.default is default
    assert default["keys"][0][1] == [1.0, 0.5, 0.2, 1.0]
