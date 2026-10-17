from infernux.engine.interaction import (
    COMMAND_PALETTE_CONTEXT_ID,
    COMMAND_PALETTE_MODAL_ID,
    CommandSource,
    CommandStatus,
    EditorCommand,
    EditorCommandRegistry,
    FocusService,
    KeyChord,
    ModalService,
    SelectionService,
    SelectionTarget,
    ShortcutBinding,
    ShortcutEvent,
    ShortcutRouteStatus,
    ShortcutRouter,
    ShortcutScope,
)
from infernux.engine.interaction.command_palette import CommandPaletteService


def _palette():
    focus = FocusService()
    selection = SelectionService()
    commands = EditorCommandRegistry(focus=focus, selection=selection)
    shortcuts = ShortcutRouter(commands, focus)
    modals = ModalService()
    palette = CommandPaletteService(commands, shortcuts, focus, modals)
    palette.register_commands()
    modals.register(
        COMMAND_PALETTE_MODAL_ID,
        is_active=lambda: palette.is_active,
        render=lambda _ctx: None,
        cancel=palette.close,
    )
    return palette, commands, shortcuts, focus, selection, modals


def test_palette_executes_against_the_context_that_opened_it():
    palette, commands, _shortcuts, focus, selection, modals = _palette()
    observed = []
    commands.register(
        EditorCommand(
            "scene.test_command",
            lambda context: observed.append(context) or True,
            display_name="Test Scene Command",
            category="Scene",
            creates_user_action=False,
        )
    )
    focus.activate_panel("scene_view", view_id="scene_view", record_history=False)
    selection.select(
        SelectionTarget.scene_object(42),
        owner_id="scene_view",
        record_history=False,
    )

    assert (
        commands.execute(
            "command_palette.open",
            source=CommandSource.SHORTCUT,
        ).status
        is CommandStatus.EXECUTED
    )
    assert COMMAND_PALETTE_CONTEXT_ID in {context.context_id for context in focus.input_contexts.ordered()}

    focus.activate_panel("project", view_id="project", record_history=False)
    selection.clear(record_history=False)
    result = palette.execute("scene.test_command")

    assert result.status is CommandStatus.EXECUTED
    assert observed[0].source is CommandSource.PALETTE
    assert observed[0].focus.active_panel_id == "scene_view"
    assert observed[0].selection.primary.target_id == "42"
    assert not palette.is_active
    assert modals.active_modal_id == ""
    assert COMMAND_PALETTE_CONTEXT_ID not in {context.context_id for context in focus.input_contexts.ordered()}


def test_palette_navigation_shortcuts_win_inside_modal_text_input():
    palette, commands, shortcuts, focus, _selection, _modals = _palette()
    for index in range(3):
        commands.register(
            EditorCommand(
                f"test.command_{index}",
                lambda _context: True,
                display_name=f"Command {index}",
                category="Test",
                creates_user_action=False,
            )
        )
    shortcuts.register(
        ShortcutBinding(
            "command_palette.next",
            KeyChord.parse("Down"),
            ShortcutScope.CHILD_CONTEXT,
            owner_id=COMMAND_PALETTE_CONTEXT_ID,
            allow_when_text_input=True,
            allow_when_modal=True,
            binding_id="palette.next",
        )
    )
    focus.activate_panel("scene_view", view_id="scene_view", record_history=False)
    assert palette.open(commands.context())

    result = shortcuts.route(
        ShortcutEvent(
            KeyChord.parse("Down"),
            text_input_active=True,
            modal_active=True,
        )
    )

    assert result.status is ShortcutRouteStatus.EXECUTED
    assert palette.selected_index == 1


# Native palette key edges are exercised in EditorShortcutInputTests.cpp.
