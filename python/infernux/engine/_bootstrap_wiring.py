"""BootstrapWiringMixin — extracted from EditorBootstrap."""
from __future__ import annotations



import os



def _native_shortcut_event(chord: str, text_input: bool, modal: bool):
    """Translate one native editor key edge with the current game-input owner."""
    from infernux.engine.interaction import KeyChord, ShortcutEvent
    from infernux.input import Input

    return ShortcutEvent(
        KeyChord.parse(chord),
        text_input_active=bool(text_input),
        modal_active=bool(modal),
        game_view_captured=bool(Input.is_game_focused()),
    )


class BootstrapWiringMixin:
    """BootstrapWiringMixin method group for EditorBootstrap."""

    @staticmethod
    def _toggle_window_target(wm, target_id: str) -> bool:
        """Toggle one registered utility window through the shared command path."""
        from infernux.engine._bootstrap_panels import (
            PERMANENT_EDITOR_WINDOW_TYPE_IDS,
        )

        target = str(target_id or "").strip()
        if not target or target not in wm.get_registered_types():
            return False
        if target in PERMANENT_EDITOR_WINDOW_TYPE_IDS:
            return False
        if wm.is_window_open(target):
            wm.close_window(target)
            return True
        return wm.open_window_from_user(
            target,
            reason="window_toggle_command",
        ) is not None

    def _register_core_editor_commands(self, wm, sfm) -> None:
        from infernux.engine.interaction import (
            ActionOrigin,
            ContinuousEditService,
            EditorCommand,
            KeyChord,
            SelectionDomain,
            ShortcutBinding,
            ShortcutScope,
        )
        from infernux.engine._bootstrap_panels import (
            PERMANENT_EDITOR_WINDOW_TYPE_IDS,
        )
        from infernux.engine.undo import UndoManager
        registry = self.interaction_core.commands
        shortcuts = self.interaction_core.shortcuts
        panel_interactions = self.interaction_core.panels
        pmm = self.engine._play_mode_manager if self.engine else None

        def _native_engine():
            getter = getattr(self.engine, "get_native_engine", None)
            return getter() if callable(getter) else None

        def _undo(_context):
            ContinuousEditService.instance().commit_all()
            manager = UndoManager.instance()
            if not manager or not manager.can_undo:
                return False
            manager.undo(defer=True)
            return True

        def _redo(_context):
            manager = UndoManager.instance()
            if not manager or not manager.can_redo:
                return False
            manager.redo(defer=True)
            return True

        def _toggle_play(_context):
            if not pmm:
                return False
            if not pmm.is_edit_mode:
                return bool(pmm.exit_play_mode())
            if not pmm.enter_play_mode():
                return False
            wm.open_window_from_user(
                "game_view",
                reason="play_mode_enter",
            )
            return True

        def _new_scene(_context):
            sfm.new_scene()
            return True

        def _scene_world(context) -> int:
            try:
                return int(context.payload.get("world_id", 0) or 0)
            except (TypeError, ValueError):
                return 0

        def _save_scene(context):
            world_id = _scene_world(context)
            if world_id <= 0:
                return False
            document_id = sfm.document_id_for_scene(world_id)
            if not document_id:
                return False
            from infernux.engine.interaction import DocumentActionStatus, DocumentRegistry
            result = DocumentRegistry.instance().request_save(document_id)
            return result.status is not DocumentActionStatus.REJECTED

        def _unload_scene(context):
            world_id = _scene_world(context)
            if world_id <= 0:
                return False
            return bool(sfm.request_unload_scene(world_id))

        def _open_scene_additive(context):
            path = str(context.payload.get("source_path", "") or "").strip()
            return bool(path and sfm.open_scene_additive(path))

        def _pause(_context):
            pmm.toggle_pause()
            return True

        def _step(_context):
            pmm.step_frame()
            return True

        def _toggle_scene_grid(_context) -> bool:
            native = _native_engine()
            if native is None:
                return False
            current = bool(native.is_show_grid())
            return self.interaction_core.view_commands.set_value(
                current,
                not current,
                lambda value: native.set_show_grid(bool(value)),
                description="Toggle Scene Grid",
                owner_view_id="scene_view",
            )

        def _is_scene_grid_visible(_context) -> bool:
            native = _native_engine()
            return bool(native is not None and native.is_show_grid())

        def _toggle_scene_gizmos(_context) -> bool:
            engine = self.engine
            if engine is None:
                return False
            current = bool(engine.is_show_gizmos())
            return self.interaction_core.view_commands.set_value(
                current,
                not current,
                lambda value: engine.set_show_gizmos(bool(value)),
                description="Toggle Scene Gizmos",
                owner_view_id="scene_view",
            )

        def _are_scene_gizmos_visible(_context) -> bool:
            engine = self.engine
            return bool(engine is not None and engine.is_show_gizmos())

        def _window_target(context) -> str:
            return str(context.payload.get("target_id", "") or "").strip()

        def _open_window(context):
            target_id = _window_target(context)
            if not target_id:
                return False
            return wm.open_window_from_user(target_id) is not None

        def _toggle_window_target(target_id: str) -> bool:
            return self._toggle_window_target(wm, target_id)

        def _toggle_window(context) -> bool:
            return _toggle_window_target(_window_target(context))

        def _reset_layout(_context):
            return wm.reset_layout() is not False

        def _open_console_entry(context) -> bool:
            try:
                uid = int(context.payload.get("uid", 0) or 0)
            except (TypeError, ValueError):
                return False
            panel = wm.open_window_from_user(
                "console",
                reason="status_bar_console",
            )
            if panel is None:
                return False
            selector = getattr(panel, "select_entry", None)
            if not callable(selector):
                return False
            selector(uid)
            return True

        def _console_source_path(context) -> str:
            return str(context.payload.get("source_path", "") or "").strip()

        def _open_console_source(context) -> bool:
            source_path = _console_source_path(context)
            if not source_path:
                return False
            try:
                source_line = max(
                    int(context.payload.get("source_line", 0) or 0),
                    0,
                )
            except (TypeError, ValueError):
                source_line = 0
            from infernux.engine.ui import project_utils

            return bool(
                project_utils.open_in_vscode(
                    source_path,
                    project_root=self.project_path,
                    line=source_line,
                )
            )

        def _owns_panel_command(context, command_id: str) -> bool:
            return panel_interactions.owns_active(context, command_id)

        def _can_panel_command(context, command_id: str) -> bool:
            return panel_interactions.can_execute_active(context, command_id)

        def _invoke_panel_command(context, command_id: str) -> bool:
            return panel_interactions.execute_active(context, command_id)

        def _prefab_target(context) -> tuple[int, str, str]:
            payload = context.payload
            try:
                object_id = int(
                    payload.get("object_id", 0)
                    or payload.get("target_id", 0)
                    or 0
                )
            except (TypeError, ValueError):
                object_id = 0
            if object_id <= 0 and context.selection.primary is not None:
                primary = context.selection.primary
                if primary.domain is SelectionDomain.SCENE_OBJECT:
                    object_id = primary.scene_object_id()
                elif primary.domain is SelectionDomain.COMPONENT:
                    object_id = primary.component_ids()[0]
            return (
                object_id,
                str(payload.get("path", "") or "").strip(),
                str(payload.get("current_path", "") or "").strip(),
            )

        def _command_origin(context) -> ActionOrigin:
            from infernux.engine.interaction import CommandSource

            return (
                ActionOrigin.AUTOMATION
                if context.source is CommandSource.AUTOMATION
                else ActionOrigin.USER
            )

        def _can_prefab_command(context, action: str) -> bool:
            object_id, path, _current_path = _prefab_target(context)
            service_action = {
                "save_as": "create",
                "select": "locate",
            }.get(action, action)
            return self.interaction_core.prefabs.can_execute(
                service_action,
                object_id=object_id,
                path=path,
            )

        def _execute_prefab_command(context, action: str) -> bool:
            object_id, path, current_path = _prefab_target(context)
            service = self.interaction_core.prefabs
            origin = _command_origin(context)
            if action == "save_as":
                return bool(
                    service.create_from_object(
                        object_id,
                        current_path,
                        origin=origin,
                    )
                )
            if action == "select":
                return service.locate(object_id=object_id, path=path)
            if action == "open":
                return service.open(object_id=object_id, path=path, origin=origin)
            if action == "exit":
                return service.exit(origin=origin)
            if action == "apply":
                return service.apply(object_id, origin=origin)
            if action == "revert":
                return service.revert(object_id, origin=origin)
            if action == "unpack":
                return service.unpack(object_id, origin=origin)
            return False

        def _can_target_panel_command(
            context,
            view_id: str,
            command_id: str,
        ) -> bool:
            return panel_interactions.can_execute_view(
                view_id,
                context,
                command_id,
            )

        def _invoke_target_panel_command(
            context,
            view_id: str,
            command_id: str,
        ) -> bool:
            return panel_interactions.execute_view(
                view_id,
                context,
                command_id,
            )

        def _clear_edit_selection(context) -> bool:
            if _owns_panel_command(context, "edit.deselect"):
                return _invoke_panel_command(context, "edit.deselect")
            return self.interaction_core.selection.clear(
                reason="edit_deselect",
                record_history=True,
            )

        def _copy_edit_selection(context, *, cut: bool):
            command_id = "edit.cut" if cut else "edit.copy"
            return _invoke_panel_command(context, command_id)

        def _paste_edit_selection(context):
            return _invoke_panel_command(context, "edit.paste")

        def _delete_edit_selection(context):
            return _invoke_panel_command(context, "edit.delete")

        def _rename_edit_selection(context):
            return _invoke_panel_command(context, "edit.rename")

        def _can_copy_edit_selection(context) -> bool:
            return _can_panel_command(context, "edit.copy")

        def _can_cut_edit_selection(context) -> bool:
            return _can_panel_command(context, "edit.cut")

        def _can_delete_edit_selection(context) -> bool:
            return _can_panel_command(context, "edit.delete")

        def _can_paste_edit_selection(context) -> bool:
            return _can_panel_command(context, "edit.paste")

        def _can_rename_edit_selection(context) -> bool:
            return _can_panel_command(context, "edit.rename")

        def _create_project_folder(context):
            return _invoke_target_panel_command(
                context, "project", "project.create_folder"
            )

        def _asset_rename_target(context) -> tuple[str, str]:
            return (
                str(context.payload.get("source_path", "") or "").strip(),
                str(context.payload.get("new_name", "") or "").strip(),
            )

        def _can_rename_project_asset(context) -> bool:
            source_path, new_name = _asset_rename_target(context)
            return self.interaction_core.project_assets.can_rename(
                source_path,
                new_name,
            )

        def _rename_project_asset(context) -> bool:
            source_path, new_name = _asset_rename_target(context)
            return bool(
                self.interaction_core.project_assets.rename(
                    source_path,
                    new_name,
                    origin=_command_origin(context),
                )
            )

        def _asset_import_target(context) -> tuple[tuple[str, ...], str]:
            raw_paths = context.payload.get("paths", ())
            if isinstance(raw_paths, str):
                raw_paths = (raw_paths,)
            paths = tuple(
                str(path).strip()
                for path in raw_paths
                if str(path or "").strip()
            )
            destination = str(
                context.payload.get("destination", "") or ""
            ).strip()
            return paths, destination

        def _can_import_external_assets(context) -> bool:
            paths, destination = _asset_import_target(context)
            return self.interaction_core.project_assets.can_import_external(
                paths,
                destination,
            )

        def _import_external_assets(context) -> bool:
            paths, destination = _asset_import_target(context)
            return bool(
                self.interaction_core.project_assets.import_external(
                    paths,
                    destination,
                    origin=_command_origin(context),
                    select_results=True,
                )
            )

        def _open_project_document(context) -> bool:
            from infernux.engine.interaction import (
                DocumentKind,
                DocumentOpenStatus,
            )

            path = str(context.payload.get("path", "") or "").strip()
            kind_name = str(
                context.payload.get("document_kind", "") or ""
            ).strip()
            if not path or not kind_name:
                return False
            try:
                kind = DocumentKind(kind_name)
            except ValueError:
                return False
            guid = ""
            asset_database = self.engine.get_asset_database() if self.engine else None
            if asset_database is not None:
                guid = str(asset_database.get_guid_from_path(path) or "")
            result = self.interaction_core.document_open.open_resource(
                kind,
                path,
                guid=guid,
                title=os.path.basename(path),
            )
            if result.status is DocumentOpenStatus.FAILED:
                return False

            panel_id = {
                DocumentKind.ANIMATION_CLIP: "animclip2d_editor",
                DocumentKind.ANIMATION_FSM: "animfsm_editor",
                DocumentKind.TIMELINE: "animtimeline_editor",
                DocumentKind.PARTICLE_GRAPH: "particle_graph_editor",
            }.get(kind)
            if not panel_id:
                return False

            # DocumentOpenService owns resource replacement and registration;
            # WindowManager owns the user-visible navigation edge. Calling
            # this after open_resource lets a newly registered dynamic panel
            # defer focus until register_gui completes, while the history
            # context already points at the opened document.
            return wm.open_window_from_user(
                panel_id,
                reason="project_open_document",
            ) is not None

        def _object_property_target(context):
            payload = context.payload
            try:
                object_id = int(payload.get("object_id", 0) or 0)
            except (TypeError, ValueError):
                return None
            property_name = str(payload.get("property", "") or "").strip()
            if object_id <= 0 or not property_name or "value" not in payload:
                return None
            return object_id, property_name, payload["value"]

        def _can_set_object_property(context) -> bool:
            target = _object_property_target(context)
            return bool(
                target is not None
                and self.interaction_core.scene_objects.can_set_object_property(*target)
            )

        def _set_object_property(context) -> bool:
            target = _object_property_target(context)
            return bool(
                target is not None
                and self.interaction_core.scene_objects.set_object_property(*target)
            )

        def _transform_targets(context):
            payload = context.payload
            object_ids = payload.get("object_ids", ())
            transforms = payload.get("transforms", ())
            if isinstance(object_ids, (str, bytes)) or isinstance(
                transforms, (str, bytes)
            ):
                return None
            try:
                return (
                    tuple(object_ids),
                    tuple(transforms),
                    str(payload.get("gesture_id", "") or "").strip(),
                )
            except TypeError:
                return None

        def _can_set_transforms(context) -> bool:
            target = _transform_targets(context)
            return bool(
                target is not None
                and self.interaction_core.scene_objects.can_set_transforms(*target)
            )

        def _set_transforms(context) -> bool:
            target = _transform_targets(context)
            return bool(
                target is not None
                and self.interaction_core.scene_objects.set_transforms(*target)
            )

        def _panel_entry(command_id, display_name, category, *, target=None, **options):
            if target is None:
                execute = lambda context: _invoke_panel_command(context, command_id)
                can_execute = lambda context: _can_panel_command(context, command_id)
            else:
                execute = lambda context: _invoke_target_panel_command(context, target, command_id)
                can_execute = lambda context: _can_target_panel_command(context, target, command_id)
            return EditorCommand(
                command_id,
                execute,
                display_name=display_name,
                category=category,
                can_execute=can_execute,
                **options,
            )

        commands = (
            EditorCommand(
                "file.new_scene",
                _new_scene,
                display_name="New Scene",
                category="File",
                default_shortcut="Ctrl+N",
            ),
            EditorCommand(
                "file.save",
                lambda _context: self.interaction_core.saving.save_focused().accepted,
                display_name="Save",
                category="File",
                default_shortcut="Ctrl+S",
            ),
            EditorCommand(
                "file.save_as",
                lambda _context: self.interaction_core.saving.save_focused(
                    save_as=True
                ).accepted,
                display_name="Save As",
                category="File",
                default_shortcut="Ctrl+Shift+S",
            ),
            EditorCommand(
                "edit.undo",
                _undo,
                display_name="Undo",
                category="Edit",
                can_execute=lambda _context: bool(
                    ContinuousEditService.instance().active_count
                    or (UndoManager.instance() and UndoManager.instance().can_undo)
                ),
                default_shortcut="Ctrl+Z",
            ),
            EditorCommand(
                "edit.redo",
                _redo,
                display_name="Redo",
                category="Edit",
                can_execute=lambda _context: bool(
                    UndoManager.instance() and UndoManager.instance().can_redo
                ),
                default_shortcut="Ctrl+Shift+Z",
            ),
            EditorCommand(
                "edit.copy",
                lambda context: _copy_edit_selection(context, cut=False),
                display_name="Copy",
                category="Edit",
                can_execute=_can_copy_edit_selection,
                default_shortcut="Ctrl+C",
            ),
            EditorCommand(
                "edit.cut",
                lambda context: _copy_edit_selection(context, cut=True),
                display_name="Cut",
                category="Edit",
                can_execute=_can_cut_edit_selection,
                default_shortcut="Ctrl+X",
            ),
            EditorCommand(
                "edit.paste",
                _paste_edit_selection,
                display_name="Paste",
                category="Edit",
                can_execute=_can_paste_edit_selection,
                default_shortcut="Ctrl+V",
            ),
            EditorCommand(
                "edit.delete",
                _delete_edit_selection,
                display_name="Delete",
                category="Edit",
                can_execute=_can_delete_edit_selection,
                default_shortcut="Delete",
            ),
            EditorCommand(
                "interaction.cancel",
                lambda _context: self.interaction_core.cancel_active_interaction(),
                display_name="Cancel Current Interaction",
                category="Edit",
                can_execute=lambda _context: self.interaction_core.can_cancel_active_interaction,
                default_shortcut="Escape",
            ),
            EditorCommand(
                "edit.deselect",
                _clear_edit_selection,
                display_name="Deselect All",
                category="Edit",
                can_execute=lambda context: bool(context.selection.targets),
                default_shortcut="Escape",
            ),
            EditorCommand(
                "edit.rename",
                _rename_edit_selection,
                display_name="Rename",
                category="Edit",
                can_execute=_can_rename_edit_selection,
                default_shortcut="F2",
            ),
            _panel_entry(
                'hierarchy.set_expanded', 'Set Hierarchy Item Expanded', 'Hierarchy'
            ),
            _panel_entry(
                'edit.duplicate', 'Duplicate', 'Edit',
                default_shortcut='Ctrl+D',
            ),
            _panel_entry(
                'view.focus_search', 'Focus Search', 'View',
                default_shortcut='Ctrl+F',
            ),
            _panel_entry(
                'component.open_script', 'Open Script', 'Component'
            ),
            _panel_entry(
                'component.copy_properties', 'Copy Component', 'Component'
            ),
            _panel_entry(
                'component.paste_properties', 'Paste Component Values', 'Component'
            ),
            _panel_entry(
                'component.paste_as_new', 'Paste Component As New', 'Component'
            ),
            _panel_entry(
                'component.remove', 'Remove Component', 'Component'
            ),
            _panel_entry(
                'component.reset', 'Reset Component', 'Component'
            ),
            _panel_entry(
                'component.move_up', 'Move Component Up', 'Component'
            ),
            _panel_entry(
                'component.move_down', 'Move Component Down', 'Component'
            ),
            _panel_entry(
                'component.reorder', 'Reorder Components', 'Component'
            ),
            _panel_entry(
                'component.add', 'Add Component', 'Component',
                target='inspector',
            ),
            _panel_entry(
                'component.set_enabled', 'Set Component Enabled', 'Component'
            ),
            EditorCommand(
                "prefab.save_as",
                lambda context: _execute_prefab_command(context, "save_as"),
                display_name="Save As Prefab",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "save_as"),
            ),
            EditorCommand(
                "prefab.select_asset",
                lambda context: _execute_prefab_command(context, "select"),
                display_name="Select Prefab Asset",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "select"),
            ),
            EditorCommand(
                "prefab.open",
                lambda context: _execute_prefab_command(context, "open"),
                display_name="Open Prefab",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "open"),
            ),
            EditorCommand(
                "prefab.apply",
                lambda context: _execute_prefab_command(context, "apply"),
                display_name="Apply Prefab Overrides",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "apply"),
            ),
            EditorCommand(
                "prefab.revert",
                lambda context: _execute_prefab_command(context, "revert"),
                display_name="Revert Prefab Overrides",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "revert"),
            ),
            EditorCommand(
                "prefab.unpack",
                lambda context: _execute_prefab_command(context, "unpack"),
                display_name="Unpack Prefab",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "unpack"),
            ),
            EditorCommand(
                "prefab.exit",
                lambda context: _execute_prefab_command(context, "exit"),
                display_name="Exit Prefab Mode",
                category="Prefab",
                can_execute=lambda context: _can_prefab_command(context, "exit"),
            ),
            EditorCommand(
                "project.create_folder",
                _create_project_folder,
                display_name="Create Folder",
                category="Assets",
                can_execute=lambda context: _can_target_panel_command(
                    context, "project", "project.create_folder"
                ),
                default_shortcut="Ctrl+Shift+N",
            ),
            _panel_entry(
                'asset.create', 'Create Asset', 'Assets',
                target='project',
            ),
            _panel_entry(
                'asset.open', 'Open Asset', 'Assets',
                target='project',
            ),
            EditorCommand(
                "asset.rename",
                _rename_project_asset,
                display_name="Rename Asset",
                category="Assets",
                can_execute=_can_rename_project_asset,
            ),
            EditorCommand(
                "asset.import_external",
                _import_external_assets,
                display_name="Import External Assets",
                category="Assets",
                can_execute=_can_import_external_assets,
            ),
            _panel_entry(
                'asset.transfer', 'Move Assets', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.reveal_in_explorer', 'Reveal in File Explorer', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.navigate_directory', 'Navigate Project', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.navigate_back', 'Navigate Project Back', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.navigate_forward', 'Navigate Project Forward', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.locate_asset', 'Locate Asset', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.set_folder_expanded', 'Set Project Folder Expanded', 'Assets',
                target='project',
            ),
            _panel_entry(
                'project.set_model_expanded', 'Set Model Contents Expanded', 'Assets',
                target='project',
            ),
            EditorCommand(
                "project.open_document",
                _open_project_document,
                display_name="Open Asset",
                category="Assets",
                can_execute=lambda context: bool(
                    context.payload.get("path")
                    and context.payload.get("document_kind")
                ),
            ),
            _panel_entry(
                'timeline.new', 'New Timeline', 'Timeline'
            ),
            _panel_entry(
                'animclip2d.new', 'New 2D Animation Clip', 'Animation'
            ),
            _panel_entry(
                'animclip2d.play_pause', 'Play / Pause 2D Animation Preview', 'Animation',
                default_shortcut='Space',
            ),
            _panel_entry(
                'animclip2d.stop', 'Stop 2D Animation Preview', 'Animation'
            ),
            _panel_entry(
                'animclip2d.previous_frame', 'Previous 2D Animation Frame', 'Animation'
            ),
            _panel_entry(
                'animclip2d.next_frame', 'Next 2D Animation Frame', 'Animation'
            ),
            _panel_entry(
                'animclip2d.clear_sequence', 'Clear 2D Animation Sequence', 'Animation'
            ),
            _panel_entry(
                'animclip2d.fill_sequence', 'Fill 2D Animation Sequence', 'Animation'
            ),
            _panel_entry(
                'animclip2d.add_frame', 'Add 2D Animation Frame', 'Animation'
            ),
            _panel_entry(
                'animfsm.new', 'New Animation State Machine', 'Animation'
            ),
            _panel_entry(
                'build.start', 'Build Player', 'Build'
            ),
            _panel_entry(
                'build.start_and_run', 'Build and Run Player', 'Build'
            ),
            _panel_entry(
                'build.cancel', 'Cancel Build', 'Build'
            ),
            _panel_entry(
                'timeline.play_pause', 'Play / Pause Timeline', 'Timeline',
                default_shortcut='Space',
            ),
            _panel_entry(
                'timeline.stop', 'Stop Timeline', 'Timeline'
            ),
            _panel_entry(
                'timeline.set_loop_preview', 'Toggle Timeline Loop Preview', 'Timeline',
                palette_visible=False,
            ),
            _panel_entry(
                'timeline.add_keyframe', 'Add Timeline Keyframe', 'Timeline'
            ),
            _panel_entry(
                'scene.create_object', 'Create GameObject', 'GameObject'
            ),
            _panel_entry(
                'scene.create_empty_parent', 'Create Empty Parent', 'GameObject'
            ),
            _panel_entry(
                'scene.instantiate_prefab', 'Instantiate Prefab', 'GameObject',
                target='hierarchy',
            ),
            _panel_entry(
                'scene.create_model', 'Create Model', 'GameObject',
                target='hierarchy',
            ),
            _panel_entry(
                'scene.rename_object', 'Rename GameObject', 'GameObject',
                target='hierarchy',
            ),
            EditorCommand(
                "scene.set_object_property",
                _set_object_property,
                display_name="Set GameObject Property",
                category="GameObject",
                can_execute=_can_set_object_property,
            ),
            EditorCommand(
                "scene.set_transforms",
                _set_transforms,
                display_name="Edit Transform",
                category="GameObject",
                can_execute=_can_set_transforms,
                # A native drag spans many rendered frames. The Transform
                # service supplies one stable gesture transaction instead of
                # wrapping every sample in a separate user action.
                creates_user_action=False,
            ),
            _panel_entry(
                'scene.move_hierarchy', 'Move GameObject', 'GameObject',
                target='hierarchy',
            ),
            _panel_entry(
                'scene.set_active', 'Set Active Scene', 'Scene',
                target='hierarchy',
            ),
            EditorCommand(
                "scene.save",
                _save_scene,
                display_name="Save Scene",
                category="Scene",
                can_execute=lambda context: _scene_world(context) > 0,
            ),
            EditorCommand(
                "scene.unload",
                _unload_scene,
                display_name="Unload Scene",
                category="Scene",
                can_execute=lambda context: _scene_world(context) > 0,
            ),
            EditorCommand(
                "scene.open_additive",
                _open_scene_additive,
                display_name="Open Scene Additive",
                category="Scene",
                can_execute=lambda context: bool(context.payload.get("source_path", "")),
            ),
            _panel_entry(
                'scene.tool.select', 'Select Tool', 'Scene',
                default_shortcut='Q',
            ),
            _panel_entry(
                'ui.nudge.left', 'Nudge UI Element Left', 'UI Editor',
                default_shortcut='Left',
            ),
            _panel_entry(
                'ui.nudge.right', 'Nudge UI Element Right', 'UI Editor',
                default_shortcut='Right',
            ),
            _panel_entry(
                'ui.nudge.up', 'Nudge UI Element Up', 'UI Editor',
                default_shortcut='Up',
            ),
            _panel_entry(
                'ui.nudge.down', 'Nudge UI Element Down', 'UI Editor',
                default_shortcut='Down',
            ),
            _panel_entry(
                'ui.nudge.left.fast', 'Nudge UI Element Left (Fast)', 'UI Editor',
                default_shortcut='Shift+Left',
            ),
            _panel_entry(
                'ui.nudge.right.fast', 'Nudge UI Element Right (Fast)', 'UI Editor',
                default_shortcut='Shift+Right',
            ),
            _panel_entry(
                'ui.nudge.up.fast', 'Nudge UI Element Up (Fast)', 'UI Editor',
                default_shortcut='Shift+Up',
            ),
            _panel_entry(
                'ui.nudge.down.fast', 'Nudge UI Element Down (Fast)', 'UI Editor',
                default_shortcut='Shift+Down',
            ),
            _panel_entry(
                'scene.tool.move', 'Move Tool', 'Scene',
                default_shortcut='W',
            ),
            _panel_entry(
                'scene.tool.rotate', 'Rotate Tool', 'Scene',
                default_shortcut='E',
            ),
            _panel_entry(
                'scene.tool.scale', 'Scale Tool', 'Scene',
                default_shortcut='R',
            ),
            _panel_entry(
                'scene.tool.rect', 'Rect Tool', 'Scene',
                default_shortcut='T',
            ),
            _panel_entry(
                'scene.align_to_camera', 'Align With View', 'Scene',
                target='scene_view',
                default_shortcut='Ctrl+Shift+F',
            ),
            _panel_entry(
                'scene.frame_selected', 'Frame Selected', 'Scene',
                target='scene_view',
                default_shortcut='F',
            ),
            EditorCommand(
                "scene.toggle_grid",
                _toggle_scene_grid,
                display_name="Toggle Scene Grid",
                category="Scene",
                can_execute=lambda _context: _native_engine() is not None,
                is_checked=_is_scene_grid_visible,
            ),
            EditorCommand(
                "scene.toggle_gizmos",
                _toggle_scene_gizmos,
                display_name="Toggle Scene Gizmos",
                category="Scene",
                can_execute=lambda _context: self.engine is not None,
                is_checked=_are_scene_gizmos_visible,
            ),
            _panel_entry(
                'scene.set_coordinate_space', 'Set Scene Coordinate Space', 'Scene'
            ),
            EditorCommand(
                "play.toggle",
                _toggle_play,
                display_name="Play / Stop",
                category="Play",
                can_execute=lambda _context: pmm is not None,
            ),
            EditorCommand(
                "play.pause",
                _pause,
                display_name="Pause / Resume",
                category="Play",
                can_execute=lambda _context: bool(pmm and pmm.is_playing),
            ),
            EditorCommand(
                "play.step",
                _step,
                display_name="Step",
                category="Play",
                can_execute=lambda _context: bool(pmm and pmm.is_paused),
            ),
            EditorCommand(
                "window.open",
                _open_window,
                display_name="Open Window",
                category="Window",
                can_execute=lambda context: _window_target(context)
                in wm.get_registered_types(),
                is_checked=lambda context: bool(
                    _window_target(context)
                    and wm.is_window_open(_window_target(context))
                ),
            ),
            EditorCommand(
                "window.toggle",
                _toggle_window,
                display_name="Toggle Window",
                category="Window",
                can_execute=lambda context: _window_target(context)
                in wm.get_registered_types()
                and _window_target(context)
                not in PERMANENT_EDITOR_WINDOW_TYPE_IDS,
                is_checked=lambda context: bool(
                    _window_target(context)
                    and wm.is_window_open(_window_target(context))
                ),
            ),
            EditorCommand(
                "window.reset_layout",
                _reset_layout,
                display_name="Reset Layout",
                category="Window",
            ),
            EditorCommand(
                "console.open_entry",
                _open_console_entry,
                display_name="Open Console Entry",
                category="Window",
                can_execute=lambda context: str(
                    context.payload.get("uid", "")
                ).strip().isdigit(),
            ),
            EditorCommand(
                "console.open_source",
                _open_console_source,
                display_name="Open Console Source",
                category="Console",
                can_execute=lambda context: bool(_console_source_path(context)),
            ),
            _panel_entry(
                'console.clear', 'Clear Console', 'Console',
                target='console',
            ),
            _panel_entry(
                'console.set_option', 'Set Console Option', 'Console',
                target='console',
            ),
            _panel_entry(
                'console.set_search', 'Search Console', 'Console',
                target='console',
            ),
            _panel_entry(
                'console.set_detail_height', 'Resize Console Details', 'Console',
                target='console',
            ),
        )
        for command in commands:
            registry.register(command, replace=True)

        # ObjectField context actions are ordinary global commands.  The
        # popup freezes its field model in the command payload, so all asset
        # drawers share one enablement and execution contract.
        from infernux.engine.interaction import register_asset_reference_commands

        register_asset_reference_commands(registry)

        # A panel descriptor is the authority for its interaction surface.
        # Project panel-only commands into the global registry so shortcuts,
        # pointer actions, menus, and automation cannot drift into separate
        # registration paths. Explicit commands above retain their richer
        # labels and destination semantics.
        for _type_id, descriptor in panel_interactions.descriptors():
            for command_spec in descriptor.commands:
                command_id = command_spec.command_id
                if registry.get(command_id) is not None:
                    continue
                registry.register(
                    EditorCommand(
                        command_id,
                        lambda context, _command_id=command_id: _invoke_panel_command(
                            context,
                            _command_id,
                        ),
                        display_name=command_id.replace(".", " ").replace("_", " ").title(),
                        category="Panel",
                        can_execute=lambda context, _command_id=command_id: _can_panel_command(
                            context,
                            _command_id,
                        ),
                        palette_visible=False,
                    )
                )

        bindings = (
            ("file.new_scene", "Ctrl+N", "default.file.new_scene"),
            ("file.save", "Ctrl+S", "default.file.save"),
            ("file.save_as", "Ctrl+Shift+S", "default.file.save_as"),
            ("edit.undo", "Ctrl+Z", "default.edit.undo"),
            ("edit.redo", "Ctrl+Shift+Z", "default.edit.redo.shift"),
            ("edit.redo", "Ctrl+Y", "default.edit.redo.y"),
            (
                "command_palette.open",
                "Ctrl+Shift+P",
                "default.command_palette.open",
            ),
            (
                "scene.align_to_camera",
                "Ctrl+Shift+F",
                "default.scene.align_to_camera",
            ),
        )
        default_shortcuts = [
            ShortcutBinding(command_id, KeyChord.parse(chord), binding_id=binding_id)
            for command_id, chord, binding_id in bindings
        ]
        default_shortcuts.append(
            ShortcutBinding(
                "interaction.cancel",
                KeyChord.parse("Escape"),
                ShortcutScope.GLOBAL,
                priority=10_000,
                allow_when_text_input=True,
                allow_when_modal=True,
                allow_when_captured=True,
                binding_id="default.interaction.cancel",
            )
        )
        default_shortcuts.extend(
            (
                ShortcutBinding(
                    "interaction.cancel",
                    KeyChord.parse("Escape"),
                    ShortcutScope.CHILD_CONTEXT,
                    owner_id=self.interaction_core.transient_interactions.CONTEXT_ID,
                    priority=10_000,
                    allow_when_text_input=True,
                    allow_when_modal=True,
                    allow_when_captured=True,
                    binding_id="default.interaction.cancel.transient",
                ),
                ShortcutBinding(
                    "command_palette.previous",
                    KeyChord.parse("Up"),
                    ShortcutScope.CHILD_CONTEXT,
                    owner_id="command_palette",
                    priority=10_000,
                    allow_when_text_input=True,
                    allow_when_modal=True,
                    binding_id="default.command_palette.previous",
                ),
                ShortcutBinding(
                    "command_palette.next",
                    KeyChord.parse("Down"),
                    ShortcutScope.CHILD_CONTEXT,
                    owner_id="command_palette",
                    priority=10_000,
                    allow_when_text_input=True,
                    allow_when_modal=True,
                    binding_id="default.command_palette.next",
                ),
                ShortcutBinding(
                    "command_palette.execute",
                    KeyChord.parse("Enter"),
                    ShortcutScope.CHILD_CONTEXT,
                    owner_id="command_palette",
                    priority=10_000,
                    allow_when_text_input=True,
                    allow_when_modal=True,
                    binding_id="default.command_palette.execute",
                ),
            )
        )
        panel_interactions.require_registered_commands(
            command.command_id for command in registry.commands
        )
        default_shortcuts.extend(panel_interactions.iter_shortcut_bindings())
        self.interaction_core.preferences.bind_shortcuts(
            default_shortcuts,
            shortcuts,
        )

    def _wire_menu_bar_callbacks(self, wm):
        """Wire C++ MenuBarPanel callbacks to Python managers."""
        mb = self.menu_bar
        sfm = self.scene_file_manager
        engine = self.engine

        self._register_core_editor_commands(wm, sfm)
        command_registry = self.interaction_core.commands
        shortcut_router = self.interaction_core.shortcuts
        from infernux.engine.interaction import (
            CommandSource,
            EditorCommand,
        )

        def _payload(argument):
            target_id = str(argument or "").strip()
            return {"target_id": target_id} if target_id else {}

        mb.execute_command = lambda command_id, source, argument: (
            command_registry.execute(
                command_id,
                source=CommandSource(source),
                payload=_payload(argument),
            ).accepted
        )
        mb.can_execute_command = lambda command_id, argument: (
            command_registry.can_execute(
                command_id,
                command_registry.context(CommandSource.MENU, _payload(argument)),
            )
        )
        mb.is_command_checked = lambda command_id, argument: (
            command_registry.is_checked(
                command_id,
                command_registry.context(CommandSource.MENU, _payload(argument)),
            )
        )
        self.shortcut_input.route_shortcut = (
            lambda chord, text_input, modal: shortcut_router.route(
                _native_shortcut_event(chord, text_input, modal)
            ).consumed
        )

        # Scene file operations
        if sfm:
            mb.on_request_close = lambda: sfm.request_close()

        # Window management
        from infernux.lib import WindowTypeInfo
        def _get_registered_types():
            types = wm.get_registered_types()
            result = []
            seen = set()
            for type_id, info in types.items():
                wti = WindowTypeInfo()
                wti.type_id = type_id
                wti.display_name = info.display_name
                wti.title_key = info.title_key or ""
                wti.menu_path = info.menu_path
                wti.menu_path_keys = list(info.menu_path_keys)
                wti.singleton = info.singleton
                result.append(wti)
                seen.add(type_id)

            # Safety net: keep core editor panels discoverable from Window menu
            # even if registration order/state gets out of sync.
            essential = {
                "inspector": "Inspector",
                "project": "Project",
            }
            for type_id, label in essential.items():
                if type_id in seen:
                    continue
                wti = WindowTypeInfo()
                wti.type_id = type_id
                wti.display_name = label
                wti.menu_path = "Window"
                wti.singleton = True
                result.append(wti)
            return result
        mb.get_registered_types = _get_registered_types
        wm.add_type_change_listener(mb.invalidate_window_type_cache)
        mb.invalidate_window_type_cache()

        # Close request from C++ engine
        native = engine.get_native_engine() if engine else None
        if native:
            mb.is_close_requested = lambda: native.is_close_requested()

        # Utility settings are normal WindowManager-owned panel surfaces. The
        # project menu only dispatches commands; it never owns their instances
        # or renders a parallel panel list.
        for command_id, display_name, panel_id in (
            ("window.toggle.build_settings", "Build Settings", "build_settings"),
            ("window.toggle.preferences", "Preferences", "preferences"),
            ("window.toggle.history", "History", "history"),
            (
                "window.toggle.physics_layers",
                "Physics Layer Matrix",
                "physics_settings",
            ),
            (
                "window.toggle.environment",
                "Environment Settings",
                "environment_settings",
            ),
        ):
            command_registry.register(
                EditorCommand(
                    command_id,
                    lambda _context, target=panel_id: self._toggle_window_target(
                        wm,
                        target,
                    ),
                    display_name=display_name,
                    category="Window",
                    is_checked=lambda _context, target=panel_id: wm.is_window_open(target),
                ),
                replace=True,
            )

        # Global confirmations are registered at overlay priority so
        # dynamically-created or undocked editors can never cover them.
        from infernux.lib import InxGUIRenderable, InxGUIContext
        from infernux.engine.ui.dirty_panel_confirmation import (
            DirtyPanelConfirmationCoordinator,
        )
        from infernux.engine.ui.project_delete_confirmation import (
            ProjectDeleteConfirmationCoordinator,
        )
        from infernux.engine.ui.external_document_conflict import (
            ExternalDocumentConflictCoordinator,
        )
        _dirty_panels = DirtyPanelConfirmationCoordinator.instance()
        _project_delete = ProjectDeleteConfirmationCoordinator.instance()
        _external_conflicts = ExternalDocumentConflictCoordinator.instance()
        from infernux.engine.ui.command_palette import CommandPalettePresenter

        self._command_palette_presenter = CommandPalettePresenter(
            self.interaction_core.command_palette,
            self.interaction_core.modals,
        )
        from infernux.engine.ui.modal_portal import ModalPortal
        _modal_portal = ModalPortal(self.interaction_core.modals)

        class _EditorGlobalOverlays(InxGUIRenderable):
            def on_render(self, ctx: InxGUIContext):
                _external_conflicts.poll()
                _modal_portal.on_render(ctx)

        self._editor_global_overlays = _EditorGlobalOverlays()
        engine.register_gui(
            "editor_global_overlays",
            self._editor_global_overlays,
            priority=1000,
        )

    def _wire_toolbar_callbacks(self, engine):
        """Wire C++ ToolbarPanel callbacks to Python PlayModeManager."""
        self._wire_toolbar_callbacks_on(self.toolbar, engine)

    def _wire_status_bar_listener(self):
        """Wire the native status bar to the shared EngineStatus state."""
        sb = self.status_bar
        from infernux.engine.i18n import t as _t

        # Fold status expiry/synchronization into the existing Python frame tick.
        # A separate Python ImGui renderable would add another C++/Python virtual
        # dispatch every frame even though status text changes rarely.
        from infernux.engine.ui.engine_status import EngineStatus

        last_status = [None]
        last_hierarchy_header = [None]

        def _sync_engine_status():
            state = EngineStatus.get()
            if state != last_status[0]:
                last_status[0] = state
                text, progress, kind = state
                sb.set_engine_status(text, progress, kind)

            sfm = self.scene_file_manager
            prefab_mode = bool(sfm and sfm.is_prefab_mode)
            scene_name = sfm.get_display_name() if sfm else ""
            prefab_name = (
                _t("hierarchy.prefab_mode_header").format(name=scene_name)
                if prefab_mode else ""
            )
            header_state = (scene_name, prefab_mode, prefab_name)
            if header_state != last_hierarchy_header[0]:
                last_hierarchy_header[0] = header_state
                self.hierarchy.set_scene_header_snapshot(*header_state)

        self.engine._editor_frame_sync_callback = _sync_engine_status
        _sync_engine_status()

    def _wire_hierarchy_callbacks(self):
        """Wire C++ HierarchyPanel callbacks to Python managers."""
        from infernux.engine.bootstrap_hierarchy import wire_hierarchy_callbacks
        wire_hierarchy_callbacks(self)
        self._wire_native_transient_interactions(self.hierarchy, "hierarchy")

    def _wire_project_callbacks(self):
        """Wire C++ ProjectPanel callbacks to Python managers."""
        from infernux.engine.bootstrap_project import wire_project_callbacks
        wire_project_callbacks(self)
        self._wire_native_transient_interactions(self.project_panel, "project")

    def _wire_inspector_callbacks(self):
        """Wire C++ InspectorPanel callbacks to Python managers."""
        from infernux.engine.bootstrap_inspector import wire_inspector_callbacks
        wire_inspector_callbacks(self)
        self._wire_native_transient_interactions(self.inspector_panel, "inspector")

    def _wire_ui_editor(self):
        ui_editor = self.ui_editor

        from infernux.engine.interaction import SelectionService

        selection = SelectionService.instance()
        previous_service = getattr(self, "_ui_editor_selection_service", None)
        previous_listener = getattr(self, "_ui_editor_selection_listener", None)
        if previous_service is not None and callable(previous_listener):
            previous_service.remove_listener(previous_listener)

        def on_global_selection_changed(change):
            self._project_ui_editor_selection(change.after)

        self._ui_editor_selection_service = selection
        self._ui_editor_selection_listener = on_global_selection_changed
        selection.add_listener(on_global_selection_changed)
        self._project_ui_editor_selection(selection.snapshot)

    def _project_ui_editor_selection(self, snapshot) -> None:
        from infernux.engine.interaction import SelectionDomain

        primary = snapshot.primary
        object_id = 0
        if primary is not None:
            if primary.domain is SelectionDomain.SCENE_OBJECT:
                object_id = primary.scene_object_id()
            elif primary.domain is SelectionDomain.COMPONENT:
                object_id = primary.component_ids()[0]

        obj = None
        if object_id:
            from infernux.lib import SceneManager

            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(object_id) if scene else None
        self.ui_editor.project_global_selection(obj)

