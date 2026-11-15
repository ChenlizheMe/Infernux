"""Present failed asset loads without changing their source files."""

from __future__ import annotations

from infernux.debug import Debug
from infernux.engine.path_utils import lexical_path_key


class AssetLoadErrors:
    _instance = None

    def __init__(self, database, project_assets):
        self.database = database
        self.project_assets = project_assets
        self.generation = -1
        self.errors = {}
        self._seen = {}
        self._path_errors = {}
        self._listeners = []
        self._published_errors = ()
        AssetLoadErrors._instance = self

    @classmethod
    def instance(cls):
        return cls._instance

    def add_listener(self, callback) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _publish_changes(self) -> None:
        current = tuple(sorted((key, str(entry.get("guid", "")), entry["error"])
                               for key, entry in self.errors.items()))
        if current == self._published_errors:
            return
        self._published_errors = current
        for callback in tuple(self._listeners):
            callback()
        from infernux.host.queries import emit_signal
        emit_signal("assets.changed")

    def report(self, path: str, error: str) -> None:
        guid = self.database.get_guid_from_path(path)
        if guid:
            self.database.report_asset_load_failure(guid, error)
        key = lexical_path_key(path)
        entry = dict(self.errors.get(key, {}))
        entry.update(guid=guid, path=path, error=error)
        self.errors[key] = entry
        if not guid:
            self._path_errors[key] = entry
        self._log(entry)
        self._publish_changes()

    def clear(self, path: str) -> None:
        """A successful import (or deletion) ends a path-only failure."""
        key = lexical_path_key(path)
        self._path_errors.pop(key, None)
        self.errors.pop(key, None)
        self._seen.pop(key, None)
        self._publish_changes()

    def _log(self, entry) -> None:
        key = lexical_path_key(entry["path"])
        stamp = entry["error"]
        if self._seen.get(key) != stamp:
            self._seen[key] = stamp
            Debug.log_error(
                f"Asset load failed for '{entry['path']}': {stamp}",
                source_file=entry["path"],
            )

    def error(self, path: str) -> str:
        # Catalog paths are canonical; Inspector lookups must not resolve them on disk again.
        entry = self.errors.get(lexical_path_key(path))
        return entry["error"] if entry is not None else ""

    def show(self, path: str) -> None:
        entry = self.errors.get(lexical_path_key(path))
        if entry is not None:
            Debug.log_error(
                f"Asset load failed for '{path}': {entry['error']}", source_file=path,
            )

    def poll(self) -> None:
        generation = self.database.query_generation
        if generation != self.generation:
            self.generation = generation
            entries = self.database.get_import_failures()
            self.errors = {**self._path_errors,
                           **{lexical_path_key(entry["path"]): entry for entry in entries}}
            self._seen = {key: stamp for key, stamp in self._seen.items() if key in self.errors}
            for entry in self.errors.values():
                self._log(entry)
            self._publish_changes()


def report_asset_load_error(path: str, error: str) -> None:
    presenter = AssetLoadErrors.instance()
    if presenter is not None:
        presenter.report(path, error)
    else:
        Debug.log_error(f"Asset load failed for '{path}': {error}")


def clear_asset_load_error(path: str) -> None:
    presenter = AssetLoadErrors.instance()
    if presenter is not None:
        presenter.clear(path)
