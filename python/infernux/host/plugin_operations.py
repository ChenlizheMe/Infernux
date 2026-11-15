"""Public plugin/theme commands delegate to the same editor authorities as UI."""
from __future__ import annotations

from .editor import EditorAutomationHost
from .operations import OperationError, OperationKind
from .operation_support import on_editor, operation


def build_plugin_operations():
    reference = {"reference": {"type": "string", "minLength": 1}}
    return (
        operation("infernux.plugin.list", OperationKind.QUERY,
                  "Read installed plugins and their actual activation/errors; no implicit update.",
                  _plugins, capability="plugin.read", tags=("plugin", "installed", "status")),
        operation("infernux.plugin.releases", OperationKind.QUERY,
                  "List compatible releases from the installed publisher without changing its pin.",
                  _releases, capability="plugin.read", input_properties=reference, required=("reference",),
                  thread="mixed", tags=("plugin", "versions", "compatible")),
        operation("infernux.plugin.install", OperationKind.WORKFLOW,
                  "Install a catalog reference or local/remote package source through PluginManager.",
                  _install, capability="plugin.write",
                  input_properties={"reference": {"type": "string", "default": ""},
                                    "source": {"type": "string", "default": ""},
                                    "selected": {"type": ["array", "null"], "items": {"type": "string"}},
                                    "install_dependencies": {"type": "boolean", "default": True}},
                  side_effects=("Installs package-owned files and dependencies, then activates current content.",),
                  thread="mixed", tags=("plugin", "install", "package")),
        operation("infernux.plugin.update", OperationKind.WORKFLOW,
                  "Overwrite publisher files with a selected/latest compatible release; preserve user-added files.",
                  _update, capability="plugin.write",
                  input_properties={**reference, "release_tag": {"type": "string", "default": ""},
                                    "source": {"type": "string", "default": ""}}, required=("reference",),
                  side_effects=("Overwrites package-owned local edits and deletes removed publisher members.",),
                  thread="mixed", tags=("plugin", "update", "overwrite")),
        operation("infernux.plugin.enabled.set", OperationKind.COMMAND,
                  "Enable or disable a plugin using its dependency and lifecycle rules.",
                  _enabled, capability="plugin.write", input_properties={**reference, "enabled": {"type": "boolean"}},
                  required=("reference", "enabled"), side_effects=("Changes package activation and its contributions.",),
                  tags=("plugin", "enable", "disable")),
        operation("infernux.plugin.uninstall", OperationKind.COMMAND,
                  "Uninstall package-owned members while preserving user-added and shared files.",
                  _uninstall, capability="plugin.write", input_properties=reference, required=("reference",),
                  side_effects=("Retires package contributions and removes its owned files/dependencies.",),
                  tags=("plugin", "uninstall", "ownership")),
        operation("infernux.plugin.reload", OperationKind.COMMAND,
                  "Reload current package content through the editor's normal reload coordinator.",
                  _reload, capability="plugin.write", input_properties=reference, required=("reference",),
                  side_effects=("Retires and reloads current plugin contributions.",), tags=("plugin", "reload")),
        operation("infernux.theme.list", OperationKind.QUERY,
                  "Read the cached editor theme catalog and actual selection.",
                  _themes, capability="theme.read", tags=("theme", "editor", "catalog")),
        operation("infernux.theme.select", OperationKind.COMMAND,
                  "Select an available editor theme and persist the personal preference.",
                  _theme_select, capability="theme.write", input_properties={"name": {"type": "string", "minLength": 1}},
                  required=("name",), side_effects=("Changes editor appearance and personal preference.",),
                  tags=("theme", "select", "editor")),
        operation("infernux.theme.refresh", OperationKind.COMMAND,
                  "Refresh themes from current enabled plugins using the theme service.",
                  _theme_refresh, capability="theme.write", side_effects=("Refreshes current theme contributions.",),
                  tags=("theme", "refresh", "editor")),
    )


def _manager():
    return on_editor("infernux.plugin.manager", lambda: EditorAutomationHost.instance().plugin_manager())


def _plugins():
    def read():
        manager = EditorAutomationHost.instance().plugin_manager()
        return {"plugins": [{"record": record,
                             "state": manager.states[record["reference"].casefold()].snapshot()
                             if record["reference"].casefold() in manager.states else None}
                            for record in manager.registry.installed()],
                "restart_pending": manager.pending_install is not None}
    return on_editor("infernux.plugin.list", read)


def _releases(reference):
    return {"releases": list(_manager().available_releases(reference))}


def _install(reference="", source="", selected=None, install_dependencies=True):
    if bool(reference.strip()) == bool(source.strip()):
        raise OperationError("operation.invalid_arguments", "Provide exactly one plugin reference or source")
    manager = _manager()
    if reference:
        staged = manager.download_reference(reference)
        if selected is None:
            return on_editor("infernux.plugin.install", lambda: manager.install_reference(
                reference, install_dependencies=install_dependencies).snapshot())
        descriptor = manager.registry.find(reference)["source"]
    else:
        staged = manager.download_source(source)
        descriptor = staged["source"]
    return on_editor("infernux.plugin.install", lambda: manager.install_package(
        str(staged["path"]), source=descriptor, selected=selected,
        install_dependencies=install_dependencies, expected_reference=reference).snapshot())


def _update(reference, release_tag="", source=""):
    if source and release_tag:
        raise OperationError("operation.invalid_arguments", "Provide a local update source or a release tag")
    manager = _manager()
    if source:
        staged = manager.download_source(source, update=True)
    else:
        if not release_tag:
            releases = manager.available_releases(reference)
            if not releases:
                raise OperationError("plugin.release_unavailable", "No compatible published release; provide an update source")
            release_tag = releases[0]["release_tag"]
        staged = manager.download_update(reference, release_tag)
    return on_editor("infernux.plugin.update", lambda: manager.install_package(
        str(staged["path"]), source=staged["source"], update=True, expected_reference=reference).snapshot())


def _enabled(reference, enabled):
    return on_editor("infernux.plugin.enabled.set", lambda: EditorAutomationHost.instance().plugin_manager()
                     .set_enabled(reference, enabled).snapshot())


def _uninstall(reference):
    return on_editor("infernux.plugin.uninstall", lambda: EditorAutomationHost.instance().plugin_manager().uninstall(reference))


def _reload(reference):
    return on_editor("infernux.plugin.reload", lambda: EditorAutomationHost.instance().plugin_manager().reload(reference).snapshot())


def _theme_service():
    from infernux.engine.editor_themes import EditorThemes
    service = EditorThemes.instance()
    if service is None:
        raise OperationError("editor.unavailable", "Editor theme service is unavailable")
    return service


def _themes():
    return on_editor("infernux.theme.list", lambda: _theme_service().snapshot())


def _theme_select(name):
    def select():
        service = _theme_service()
        service.select(name)
        return service.snapshot()
    return on_editor("infernux.theme.select", select)


def _theme_refresh():
    def refresh():
        service = _theme_service()
        service.refresh_all()
        return service.snapshot()
    return on_editor("infernux.theme.refresh", refresh)
