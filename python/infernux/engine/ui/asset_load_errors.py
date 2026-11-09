"""Present failed asset loads without changing their source files."""

from __future__ import annotations

from infernux.debug import Debug
from infernux.engine.path_utils import is_path_within, lexical_path_key, relative_path
from .project_delete_confirmation import ProjectDeleteConfirmationCoordinator


class AssetLoadErrors:
    _instance = None

    def __init__(self, database, project_assets):
        self.database = database
        self.project_assets = project_assets
        self.generation = -1
        self.errors = {}
        self._seen = {}
        self._pending = {}
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
        Debug.log_error(f"Asset load failed for '{path}': {error}")

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
            self.errors = {lexical_path_key(entry["path"]): entry for entry in entries}
            current = {entry["guid"] for entry in entries}
            self._pending = {guid: entry for guid, entry in self._pending.items() if guid in current}
            self._seen = {guid: stamp for guid, stamp in self._seen.items() if guid in current}
            for entry in entries:
                stamp = (entry["modified_ns"], entry["size"], entry["error"])
                if not entry["path"].casefold().endswith(".scene") and self._seen.get(entry["guid"]) != stamp:
                    self._pending[entry["guid"]] = entry
        if not self._pending:
            return
        coordinator = ProjectDeleteConfirmationCoordinator.instance()
        if coordinator.is_active:
            return
        entry = min(self._pending.values(), key=lambda item: item["path"])
        path = entry["path"]
        root = self.database.project_root
        display_path = relative_path(path, root) if is_path_within(path, root) else path
        accepted = coordinator.request(
            (path,),
            lambda paths: bool(self.project_assets.delete(paths)),
            load_error=f"{display_path}\n{entry['error']}",
            load_error_is_current=lambda: bool(self.database.get_import_error(entry["guid"])),
        )
        if accepted:
            self._pending.pop(entry["guid"])
            self._seen[entry["guid"]] = (entry["modified_ns"], entry["size"], entry["error"])


def report_asset_load_error(path: str, error: str) -> None:
    presenter = AssetLoadErrors.instance()
    if presenter is not None:
        presenter.report(path, error)
    else:
        Debug.log_error(f"Asset load failed for '{path}': {error}")
