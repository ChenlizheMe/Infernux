"""Present failed asset loads without changing their source files."""

from __future__ import annotations

import os

from infernux.debug import Debug, DebugConsole
from infernux.engine.path_utils import is_path_within, lexical_path_key, relative_path
from .editor_modal import EditorModalAction, begin_editor_modal, end_editor_modal, render_editor_modal_actions
from infernux.engine.i18n import t


class _AssetLoadErrorDialog:
    """An import failure owns recovery actions, separate from asset deletion."""

    MODAL_ID = "asset.load_error"

    def __init__(self, presenter):
        from infernux.engine.interaction import EditorInteractionCore
        self.presenter = presenter
        self.modals = EditorInteractionCore.instance().modals
        self.entry = None
        self.requested = False
        self.modals.register(self.MODAL_ID, is_active=lambda: self.entry is not None,
                             render=self.render, cancel=self.cancel)

    def request(self, entry):
        if self.entry is not None or not self.modals.activate(self.MODAL_ID, owner_id="project"):
            return False
        self.entry = entry
        self.requested = True
        return True

    def cancel(self):
        self.entry = None
        self.requested = False
        self.modals.deactivate(self.MODAL_ID)

    def close(self, ctx):
        ctx.close_current_popup()
        self.cancel()

    def render(self, ctx):
        entry = self.entry
        title = t("asset.load_error_title")
        if not begin_editor_modal(ctx, popup_id=f"{title}###asset_load_error", title=title,
                                   semantic_id="asset.load_error.dialog", request_open=self.requested,
                                   width=680.0, height=280.0):
            self.requested = False
            return
        self.requested = False
        try:
            if not self.presenter.database.get_import_error(entry["guid"]):
                self.close(ctx)
                return
            from .igui import IGUI
            ctx.text_wrapped(t("asset.load_error_message"))
            root = self.presenter.database.project_root
            path = relative_path(entry["path"], root) if is_path_within(entry["path"], root) else entry["path"]
            with IGUI.mono_font(ctx):
                ctx.text_wrapped(path)
            ctx.text_wrapped(entry["error"])
            render_editor_modal_actions(ctx, (
                EditorModalAction(t("editor.modal.delete"), "delete", lambda: self.delete(ctx), kind="danger"),
                EditorModalAction(t("asset.load_error_reveal"), "reveal",
                                  lambda: self.presenter.project_assets.reveal(entry["path"]), kind="ghost"),
                EditorModalAction(t("asset.load_error_ignore"), "ignore", lambda: self.close(ctx), kind="ghost"),
                EditorModalAction(t("asset.load_error_retry"), "retry", lambda: self.retry(ctx), kind="primary"),
            ), semantic_prefix="asset.load_error")
        finally:
            end_editor_modal(ctx)

    def retry(self, ctx):
        from infernux.core.assets import AssetManager
        result = AssetManager.reimport_asset(self.entry["path"], database=self.presenter.database)
        if result:
            self.close(ctx)
        else:
            self.entry["error"] = self.presenter.database.get_import_error(self.entry["guid"])

    def delete(self, ctx):
        if self.presenter.project_assets.delete((self.entry["path"],)):
            self.close(ctx)


class AssetLoadErrors:
    _instance = None

    def __init__(self, database, project_assets):
        self.database = database
        self.project_assets = project_assets
        self.generation = -1
        self.errors = {}
        self._seen = {}
        self._pending = {}
        self._dialog = None
        AssetLoadErrors._instance = self

    @classmethod
    def instance(cls):
        return cls._instance

    def report(self, path: str, error: str) -> None:
        guid = self.database.get_guid_from_path(path)
        if not guid:
            return
        self.database.report_asset_load_failure(guid, error)
        key = lexical_path_key(path)
        entry = dict(self.errors.get(key, {}))
        entry.update(guid=guid, path=path, error=error)
        self.errors[key] = entry
        if not path.casefold().endswith(".scene"):
            self._pending[guid] = entry
        self._publish_diagnostic(entry)

    @staticmethod
    def _publish_diagnostic(entry) -> None:
        path, error = entry["path"], entry["error"]
        Debug.set_diagnostic("asset.import", lexical_path_key(path),
                             f"Cannot load {os.path.basename(path)}",
                             source_file=path, details=error)

    def error(self, path: str) -> str:
        # Catalog paths are canonical; Inspector lookups must not resolve them on disk again.
        entry = self.errors.get(lexical_path_key(path))
        return entry["error"] if entry is not None else ""

    def show(self, path: str) -> None:
        entry = self.errors.get(lexical_path_key(path))
        if entry is not None:
            self._pending[entry["guid"]] = entry

    def poll(self) -> None:
        generation = self.database.query_generation
        if generation != self.generation:
            self.generation = generation
            entries = self.database.get_import_failures()
            current = {entry["guid"] for entry in entries}
            errors = {lexical_path_key(entry["path"]): entry for entry in entries}
            for entry in DebugConsole.instance().get_diagnostics("asset.import"):
                if entry.diagnostic_key not in errors:
                    Debug.resolve_diagnostic("asset.import", entry.diagnostic_key)
            self.errors = errors
            for entry in entries:
                self._publish_diagnostic(entry)
            self._pending = {guid: entry for guid, entry in self._pending.items() if guid in current}
            self._seen = {guid: stamp for guid, stamp in self._seen.items() if guid in current}
            for entry in entries:
                stamp = (entry["modified_ns"], entry["size"], entry["error"])
                if not entry["path"].casefold().endswith(".scene") and self._seen.get(entry["guid"]) != stamp:
                    self._pending[entry["guid"]] = entry
        if not self._pending:
            return
        if self._dialog is None:
            self._dialog = _AssetLoadErrorDialog(self)
        entry = min(self._pending.values(), key=lambda item: item["path"])
        path = entry["path"]
        accepted = self._dialog.request(entry)
        if accepted:
            self._pending.pop(entry["guid"])
            self._seen[entry["guid"]] = (entry["modified_ns"], entry["size"], entry["error"])


def report_asset_load_error(path: str, error: str) -> None:
    presenter = AssetLoadErrors.instance()
    if presenter is not None:
        presenter.report(path, error)
    else:
        AssetLoadErrors._publish_diagnostic({"path": path, "error": error})
