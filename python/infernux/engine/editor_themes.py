"""Cached theme contributions on the ordinary plugin/resource lifecycle."""
from __future__ import annotations

import json
from pathlib import Path

from infernux.debug import Debug
from infernux.engine.path_utils import path_key, resolved_path
from infernux.engine.preferences_store import PreferencesStore
from infernux.engine.ui.theme import Theme, active_editor_theme, list_editor_themes, _set_editor_theme_native, on_theme_changed
from infernux import lib as native
from infernux.theme_data import validate_theme_document

_BUILTINS = (
    ("infernux", "Infernux", "The built-in editor theme."),
    ("graphite", "Graphite", "A steel blue editor accent."),
    ("amber", "Amber", "A warm amber editor accent."),
)


class EditorThemes:
    _instance = None

    def __init__(self, project_root, registry, *, guid_paths=None):
        # The native editor has one active theme table. Replacing its project
        # authority retires the previous contributions and listener together.
        if EditorThemes._instance is not None:
            EditorThemes._instance.close()
        self.root = Path(resolved_path(project_root))
        self.registry = registry
        self._guid_paths = guid_paths
        self.entries = {}
        self.errors = {}
        self.owners = set()
        self._contributions = {}
        self._package_rows = {}
        self._package_errors = {}
        self._source_owners = {}
        self._tokens = {"colors": tuple(native.get_editor_theme_colors()),
                        "vec2s": tuple(native.get_editor_theme_vec2s()),
                        "floats": tuple(native.get_editor_theme_floats())}
        selected = PreferencesStore().get("editor_theme", "infernux")
        self.selection = selected if isinstance(selected, str) and selected else "infernux"
        self.revision = 0
        self._catalog = ()
        self._publishing = False
        EditorThemes._instance = self
        self._rebuild_catalog()
        self._remove_theme_listener = on_theme_changed(self._on_active_changed)

    @classmethod
    def instance(cls):
        return cls._instance

    def catalog(self):
        """Cached presentation rows, treated as read-only by consumers."""
        return self._catalog

    def snapshot(self):
        return {"themes": list(self._catalog), "revision": self.revision,
                "active": active_editor_theme(), "selection": self.selection}

    def select(self, name):
        if not isinstance(name, str) or name not in list_editor_themes():
            raise KeyError(f"Theme is unavailable: {name}")
        changed = self.selection != name or active_editor_theme() != name
        self._publishing = True
        try:
            _set_editor_theme_native(name)
        finally:
            self._publishing = False
        if self.selection != name:
            PreferencesStore().set("editor_theme", name)
            self.selection = name
        self._rebuild_catalog()
        return changed

    @staticmethod
    def _owner(reference):
        return "theme:" + reference.casefold()

    def _on_active_changed(self):
        if not self._publishing:
            self._rebuild_catalog()

    def _replace(self, owner, definitions, preferred):
        self._publishing = True
        try:
            native.replace_editor_themes(owner, definitions, preferred)
            Theme.refresh()
        finally:
            self._publishing = False

    def _rebuild_catalog(self, *, content_changed=False):
        active = active_editor_theme()
        rows = [dict(id=theme_id, name=name, author="Infernux", description=description,
                     reference="", version="", path="", available=True, error="", builtin=True)
                for theme_id, name, description in _BUILTINS]
        for owner in sorted(self._package_rows):
            rows.extend(self._package_rows[owner])
        if self.selection not in {row["id"] for row in rows}:
            rows.append(dict(id=self.selection, name=self.selection, author="", description="",
                             reference=self.selection.rsplit("/", 1)[0] if "/" in self.selection else "",
                             version="", path="", available=False,
                             error="The selected theme is unavailable or its plugin is disabled.", builtin=False))
        catalog = tuple(dict(row, active=row["id"] == active,
                             preferred=row["id"] == self.selection,
                             label=row["name"], package_reference=row["reference"]) for row in rows)
        if catalog != self._catalog or content_changed:
            self._catalog = catalog
            self.revision += 1
            from infernux.host.queries import emit_signal
            emit_signal("themes.changed", {"revision": self.revision, "active": active, "selection": self.selection})

    def _theme_paths(self, reference, record):
        folder = self.root / "Packages" / reference / "editor" / "themes"
        paths = {path_key(path): path for path in folder.glob("*.theme.json")}
        if self._guid_paths is not None:
            index = self._guid_paths()
            for member in record.get("files", ()):
                logical = str(member.get("logical_path", "")).replace("\\", "/")
                if logical.startswith("editor/themes/") and logical.lower().endswith(".theme.json"):
                    path = index.get(str(member.get("guid", "")).casefold())
                    if path and str(path).lower().endswith(".theme.json"):
                        paths[path_key(path)] = Path(path)
        return sorted(paths.values(), key=path_key)

    def refresh_package(self, reference):
        record = self.registry.installed_record(reference)
        if record is None or not bool(record.get("enabled", True)):
            return self.remove_package(reference)
        reference = str(record["reference"])
        owner = self._owner(reference)
        contributions, rows, errors = {}, {}, {}
        paths = self._theme_paths(reference, record)
        for key, value in tuple(self._source_owners.items()):
            if value == owner:
                del self._source_owners[key]
        duplicates = set()
        for path in paths:
            source = resolved_path(path)
            self._source_owners[path_key(path)] = owner
            row = dict(id=reference + "/@file:" + path.name, name=path.name,
                       author="", description="", reference=reference,
                       version=str(record.get("version", "")), path=source,
                       available=False, error="", builtin=False)
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("id"), str):
                    candidate = raw["id"]
                    if candidate and "/" not in candidate and "\\" not in candidate:
                        row["id"] = reference + "/" + candidate
                document = validate_theme_document(raw, self._tokens)
                theme_id = reference + "/" + document["id"]
                row.update(id=theme_id, name=document["name"], author=document["author"],
                           description=document["description"])
                if theme_id in rows or theme_id in duplicates:
                    duplicates.add(theme_id)
                    previous = rows.get(theme_id)
                    if previous is not None:
                        errors[previous["path"]] = f"Duplicate theme id: {theme_id}"
                    raise ValueError(f"Duplicate theme id: {theme_id}")
                contributions[theme_id] = {"name": theme_id, "colors": document["colors"],
                                           "vec2s": document["vec2s"], "floats": document["floats"]}
                row["available"] = True
            except Exception as exc:
                row["error"] = str(exc)
                errors[source] = str(exc)
            if row["id"] in rows:
                duplicates.add(row["id"])
                errors[rows[row["id"]]["path"]] = f"Duplicate theme id: {row['id']}"
                errors.setdefault(source, f"Duplicate theme id: {row['id']}")
            else:
                rows[row["id"]] = row
        for theme_id in duplicates:
            contributions.pop(theme_id, None)
            rows[theme_id].update(available=False, error=f"Duplicate theme id: {theme_id}")
        content_changed = self._contributions.get(owner, {}) != contributions
        if content_changed:
            self._replace(owner, list(contributions.values()), self.selection)
            self._contributions[owner] = contributions
        old_errors = self._package_errors.get(owner, {})
        for path, error in errors.items():
            if old_errors.get(path) != error:
                Debug.log_error(f"Theme import failed: {error}", source_file=path)
        self._package_rows[owner] = tuple(rows.values())
        self._package_errors[owner] = errors
        self.owners.add(owner)
        self._rebuild_indexes()
        self._rebuild_catalog(content_changed=content_changed)
        return content_changed

    def _rebuild_indexes(self):
        self.entries = {row["id"]: dict(row, owner=owner)
                        for owner, rows in self._package_rows.items() for row in rows if row["available"]}
        self.errors = {path: error for errors in self._package_errors.values() for path, error in errors.items()}

    def package_errors(self, reference):
        return tuple(f"Theme {path}: {error}" for path, error in self._package_errors.get(self._owner(reference), {}).items())

    def refresh_all(self):
        records = self.registry.installed()
        present = {self._owner(str(record["reference"])) for record in records}
        for owner in tuple(self.owners - present):
            self.remove_package(owner[len("theme:"):])
        for record in records:
            self.refresh_package(str(record["reference"]))
        selected = self.selection if self.selection in list_editor_themes() else "infernux"
        _set_editor_theme_native(selected)
        self._rebuild_catalog()

    def resource_changed(self, path):
        key = path_key(path)
        known_owner = self._source_owners.get(key)
        records = self.registry.installed()
        if known_owner:
            record = next((item for item in records if self._owner(str(item["reference"])) == known_owner), None)
            if record:
                self.refresh_package(str(record["reference"]))
                return
        target = Path(resolved_path(path))
        for record in sorted(records, key=lambda item: len(str(item["reference"])), reverse=True):
            reference = str(record["reference"])
            if target.is_relative_to(self.root / "Packages" / reference):
                self.refresh_package(reference)
                return

    def remove_package(self, reference):
        owner = self._owner(reference)
        contributions = self._contributions.pop(owner, {})
        existed = bool(contributions or self._package_rows.get(owner))
        if contributions:
            self._replace(owner, [], self.selection)
        self.owners.discard(owner)
        self._package_rows.pop(owner, None)
        self._package_errors.pop(owner, None)
        self._source_owners = {path: value for path, value in self._source_owners.items() if value != owner}
        self._rebuild_indexes()
        self._rebuild_catalog()
        return existed

    def close(self):
        self._remove_theme_listener()
        for owner, contributions in tuple(self._contributions.items()):
            if contributions:
                native.replace_editor_themes(owner, [], "")
        Theme.refresh()
        self.entries.clear()
        self.errors.clear()
        self.owners.clear()
        self._contributions.clear()
        self._package_rows.clear()
        self._package_errors.clear()
        self._source_owners.clear()
        if EditorThemes._instance is self:
            EditorThemes._instance = None
