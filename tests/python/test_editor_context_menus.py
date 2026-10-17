from infernux.engine.interaction import (
    CommandStatus,
    CommandSource,
    ContextMenuBuilder,
    ContextMenuCommand,
    ContextMenuSubmenu,
    EditorCommand,
    EditorCommandRegistry,
    FocusService,
    SelectionService,
    SelectionTarget,
)


class _MenuContext:
    def __init__(self, click_label: str = "") -> None:
        self.click_label = click_label
        self.items = []
        self.separators = 0
        self.closed = 0
        self.submenus = []
        self.ended_submenus = 0
        self.semantics = []

    def menu_item(self, label, shortcut, selected, enabled):
        self.items.append((label, shortcut, selected, enabled))
        return enabled and label == self.click_label

    def separator(self):
        self.separators += 1

    def close_current_popup(self):
        self.closed += 1

    def begin_menu(self, label, enabled=True, semantic_id=""):
        self.submenus.append((label, enabled, semantic_id))
        return enabled

    def end_menu(self):
        self.ended_submenus += 1

    def record_semantic_item(self, kind, label, enabled, semantic_id):
        self.semantics.append((kind, label, enabled, semantic_id))


def _registry() -> EditorCommandRegistry:
    return EditorCommandRegistry(
        focus=FocusService(),
        selection=SelectionService(),
    )


def test_graph_menu_labels_follow_the_current_locale():
    from infernux.engine.i18n import get_locale, set_locale

    registry = _registry()
    commands = ("edit.copy", "edit.cut", "edit.paste", "edit.duplicate", "edit.delete", "graph.center_view")
    for command in commands:
        registry.register(EditorCommand(command, lambda context: True, display_name=command))
    builder = ContextMenuBuilder(registry)
    specs = tuple(ContextMenuCommand(command) for command in commands)
    previous = get_locale()
    try:
        set_locale("zh")
        assert [item.label for item in builder.resolve(specs)] == ["复制", "剪切", "粘贴", "创建副本", "删除", "居中视图"]
        set_locale("en")
        assert [item.label for item in builder.resolve(specs)][:3] == ["Copy", "Cut", "Paste"]
    finally:
        set_locale(previous)


def test_deferred_context_menu_executes_the_frozen_focus_and_selection():
    focus = FocusService()
    selection = SelectionService()
    registry = EditorCommandRegistry(focus=focus, selection=selection)
    captured = []
    registry.register(
        EditorCommand(
            "asset.delete",
            lambda context: captured.append(context) or True,
            display_name="Delete",
        )
    )
    focus.activate_panel(
        "project",
        view_id="project",
        record_history=False,
    )
    selection.select(
        SelectionTarget.asset("C:/Project/Assets/A.mat"),
        owner_id="project",
        record_history=False,
    )
    builder = ContextMenuBuilder(registry)
    request = builder.render_deferred(
        _MenuContext("Delete"),
        (ContextMenuCommand("asset.delete"),),
        payload={"target_id": "C:/Project/Assets/A.mat"},
    )
    assert request is not None

    focus.activate_panel(
        "hierarchy",
        view_id="hierarchy",
        record_history=False,
    )
    selection.select(
        SelectionTarget.scene_object(42),
        owner_id="hierarchy",
        record_history=False,
    )

    result = builder.execute_resolved(request)

    assert result.result.status is CommandStatus.EXECUTED
    assert len(captured) == 1
    assert captured[0].source is CommandSource.CONTEXT_MENU
    assert captured[0].focus.active_panel_id == "project"
    assert captured[0].selection.primary == SelectionTarget.asset("C:/Project/Assets/A.mat")
    assert captured[0].payload["target_id"].endswith("A.mat")


def _flatten_menu_entries(entries):
    flattened = []
    for entry in entries:
        if isinstance(entry, ContextMenuSubmenu):
            flattened.extend(_flatten_menu_entries(entry.entries))
        else:
            flattened.append(entry)
    return flattened
