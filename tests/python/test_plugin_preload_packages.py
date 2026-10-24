from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from infernux.engine.path_utils import path_key
from infernux.plugins import (
    PluginManager,
    PluginRegistry,
)
from infernux.plugins.project_index import project_guid_paths
from test_inxpackage_plugins import (
    _export,
    _meta,
    _project,
    _source,
    _fake_inxpack as _fake_inxpack,
)


def test_package_preload_supports_relative_imports(tmp_path):
    source = _source(tmp_path / "source", "vendor/relative-preload")
    package_module = source / "editor" / "infernux_relative_preload"
    package_module.mkdir(parents=True)
    (package_module / "__init__.py").write_text("", encoding="utf-8")
    (package_module / "service.py").write_text("VALUE = 'relative-import-ready'\n", encoding="utf-8")
    (package_module / "lifecycle.py").write_text(
        "from pathlib import Path\n"
        "from infernux.lifecycle import InxPreload\n"
        "from .service import VALUE\n"
        "class RelativePreload(InxPreload):\n"
        "    def preload(self, context):\n"
        "        Path(context.project_root, 'relative-preload.txt').write_text(VALUE)\n",
        encoding="utf-8",
    )
    package = _export(source, tmp_path / "relative-preload.inxpkg")
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))

    state = manager.install_package(str(package), install_dependencies=False)

    assert state.loaded
    assert state.error == ""
    assert (project / "relative-preload.txt").read_text(encoding="utf-8") == ("relative-import-ready")


def test_editor_translation_catalog_hot_reload_is_atomic(tmp_path, monkeypatch):
    from infernux.engine import i18n

    monkeypatch.setattr(i18n, "_current_locale", "en")
    monkeypatch.setattr(i18n, "_translation_owners", {})
    monkeypatch.setattr(i18n, "_contributed_tables", {})
    source = _source(tmp_path / "source", "vendor/live-translations")
    editor = source / "editor"
    editor.mkdir()
    catalog = editor / "translations.json"

    def write_catalog(value: str) -> None:
        catalog.write_text(
            json.dumps(
                {
                    "$schema": "infernux.editor_translations",
                    "locales": {
                        "en": {"vendor.live.menu": value},
                        "zh": {"vendor.live.menu": f"zh-{value}"},
                    },
                }
            ),
            encoding="utf-8",
        )

    write_catalog("First")
    (editor / "lifecycle.py").write_text(
        "from infernux.lifecycle import InxPreload\n"
        "class LiveTranslationsPreload(InxPreload):\n"
        "    def preload(self, context):\n"
        "        pass\n",
        encoding="utf-8",
    )
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))
    state = manager.install_package(
        str(_export(source, tmp_path / "live-translations.inxpkg")),
        install_dependencies=False,
    )
    assert state.loaded, state.error
    installed_catalog = project / "Packages/vendor/live-translations/editor/translations.json"
    assert i18n.t("vendor.live.menu") == "First"
    live_initial = next(
        item for item in manager.preloads.states.values() if item.package_reference == "vendor/live-translations"
    )

    document = json.loads(installed_catalog.read_text(encoding="utf-8"))
    document["locales"]["en"]["vendor.live.menu"] = "Second"
    document["locales"]["zh"]["vendor.live.menu"] = "zh-Second"
    installed_catalog.write_text(json.dumps(document), encoding="utf-8")
    manager._on_script_catalog_changed(str(installed_catalog), "modified")
    assert i18n.t("vendor.live.menu") == "Second"

    live_before = next(
        item for item in manager.preloads.states.values() if item.package_reference == "vendor/live-translations"
    )
    assert live_before is live_initial
    installed_catalog.write_text("{", encoding="utf-8")
    manager._on_script_catalog_changed(str(installed_catalog), "modified")
    live_after = next(
        item for item in manager.preloads.states.values() if item.package_reference == "vendor/live-translations"
    )
    assert live_after is live_before
    assert live_after.loaded
    assert i18n.t("vendor.live.menu") == "Second"
    assert "JSONDecodeError" in manager.preloads.failures[path_key(installed_catalog)]


@pytest.mark.parametrize("outside_package", [False, True])
def test_project_package_destination_rejects_redirected_role(tmp_path, outside_package, directory_junction):
    from infernux.plugins.package import package_destination

    project = _project(tmp_path / "project")
    root = project / "Packages/vendor/redirected"
    root.mkdir(parents=True)
    target = tmp_path / "outside" if outside_package else root / "another_directory"
    target.mkdir()
    if os.name == "nt":
        # Exercise Windows directory redirection without requiring symbolic-link
        # privileges. This checks the destination boundary, not symlink APIs.
        directory_junction(root / "Editor", target)
    else:
        (root / "Editor").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        package_destination("vendor/redirected", "editor/new.py", project_root=str(project))
    assert not list(target.iterdir())


@pytest.mark.parametrize("fail_preload", [False, True])
def test_preload_library_ownership_follows_success_and_failure(tmp_path, fail_preload):
    from infernux.engine.project_context import is_project_component_script

    project = _project(tmp_path / "project")
    package = _source(project / "Packages/vendor/compiler", "vendor/compiler")
    library = package / "runtime/provider"
    library.mkdir(parents=True)
    implementation = library / "backend.py"
    implementation.write_text("from contextlib import contextmanager\n", encoding="utf-8")
    author = package / "runtime/AuthorComponent.py"
    author.write_text("", encoding="utf-8")
    (library / "lifecycle.py").write_text(
        "from infernux.lifecycle import InxPreload\n"
        "class LibraryPreload(InxPreload):\n"
        "    def preload(self, context):\n"
        "        context.own_python_library('runtime/provider')\n"
        + ("        raise RuntimeError('initialization failed')\n" if fail_preload else "")
        + "    def unload(self): pass\n",
        encoding="utf-8",
    )
    manager = PluginManager.startup(str(project))
    try:
        state = next(iter(manager.preloads.states.values()))
        assert state.loaded is not fail_preload, state.error
        assert is_project_component_script(str(implementation), str(project)) is fail_preload
        assert is_project_component_script(str(author), str(project))
    finally:
        manager.shutdown()
    assert is_project_component_script(str(implementation), str(project))


def test_runtime_components_follow_owned_library_install_enable_reload_and_catalog(tmp_path, monkeypatch):
    from infernux.engine.candidate_import import CandidateImportTransaction
    from infernux.engine.resources_manager import ResourcesManager
    from infernux.engine.script_dependency_graph import ScriptDependencyGraph
    from infernux.engine.project_context import (
        get_project_root, get_script_module_name, is_preload_python_library, set_project_root,
    )
    from infernux.runtime_services import _dispatch_updates, _update_snapshot

    source = _source(tmp_path / "source", "vendor/owned-runtime")
    library = source / "runtime/owned_reenable"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "API = SimpleNamespace(ticks=0)\n"
        "def advance(delta): API.ticks += 1\n", encoding="utf-8",
    )
    (source / "runtime/Startup.py").write_text(
        "from infernux.lifecycle import InxPreload\n"
        "class Startup(InxPreload):\n"
        "    def preload(self, context):\n"
        "        context.own_python_library('runtime/owned_reenable')\n"
        "        import owned_reenable\n"
        "        context.add_update(owned_reenable.advance)\n", encoding="utf-8",
    )
    (source / "runtime/Author.py").write_text("from owned_reenable import API\n", encoding="utf-8")
    project = _project(tmp_path / "project")
    consumer = project / "Assets/Consumer.py"
    consumer.write_text("from owned_reenable import API\n", encoding="utf-8")
    dependent = project / "Assets/Dependent.py"
    dependent.write_text("from Consumer import API\n", encoding="utf-8")
    unrelated = project / "Assets/Unrelated.py"
    unrelated.write_text("VALUE = 1\n", encoding="utf-8")
    graph = ScriptDependencyGraph(str(project))
    graph.index_assets()
    root = project / "Packages/vendor/owned-runtime/runtime"
    publications = []
    previous_updates = _update_snapshot()["update"]
    unrelated_module = SimpleNamespace(__file__=str(unrelated), VALUE=1)
    monkeypatch.setitem(sys.modules, "Unrelated", unrelated_module)

    class Resources:
        _project_path = str(project.resolve())
        _engine = SimpleNamespace(get_asset_database=lambda: None)
        _startup_prepared = False
        _frontend_worker_running = True
        script_library_dependents = ResourcesManager.script_library_dependents
        defer_script_catalog_publication = ResourcesManager.defer_script_catalog_publication
        process_pending_reloads = ResourcesManager.process_pending_reloads

        def __init__(self):
            self._script_catalog_publications = {}
            self.catalog_pending = False
            self.in_catalog = False
            self._event_handler = SimpleNamespace(
                _dependency_graph=graph, process_script_worker=lambda: None,
                process_pending_reloads=self.publish,
            )

        def begin_script_transaction(self, paths):
            assert not self.in_catalog
            self.paths = paths
            assert is_preload_python_library(str(root / "owned_reenable"))
            assert root / "owned_reenable/__init__.py" not in map(Path, paths)
            assert {consumer, dependent}.issubset(map(Path, paths))
            assert unrelated not in map(Path, paths)
            return "owned-runtime"

        def submit_script_change(self, path, **kwargs):
            assert kwargs["catalog_event"] is None
            return object()

        def publish(self, *, force):
            if self.catalog_pending:
                self.catalog_pending = False
                self.in_catalog = True
                manager._on_script_catalog_changed(str(root / "Startup.py"), "modified")
                assert len(publications) == 3
                self.in_catalog = False
                return 1
            broker = CandidateImportTransaction()
            name = get_script_module_name(str(root / "Author.py"), str(project))
            for path in self.paths:
                broker.register(get_script_module_name(path, str(project)), path)
            try:
                api = broker.load(name).API
                assert api is sys.modules["owned_reenable"].API
                assert broker.load("Dependent").API is api
                publications.append(api)
            finally:
                broker.rollback()
            return 1

        def retire_script_paths(self, paths):
            pass

    resources = Resources()
    monkeypatch.setattr("infernux.engine.resources_manager.ResourcesManager.instance",
                        classmethod(lambda _cls: resources))
    previous = get_project_root()
    set_project_root(str(project))
    manager = PluginManager(str(project))
    try:
        reference = "vendor/owned-runtime"
        assert manager.install_package(str(_export(source, tmp_path / "owned.inxpkg")),
                                       install_dependencies=False).loaded
        manager.set_enabled(reference, False)
        assert "owned_reenable" not in sys.modules
        assert manager.set_enabled(reference, True).loaded
        assert manager.reload(reference).loaded
        assert len(publications) == 3
        assert len({id(api) for api in publications}) == 3
        old_api = publications[-1]
        _dispatch_updates(_update_snapshot()["update"], 0.1, 0.1)
        assert old_api.ticks == 1
        startup = root / "Startup.py"
        startup.write_text(startup.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        resources.catalog_pending = True
        resources.process_pending_reloads(force=True)
        assert len(publications) == 4
        assert len({id(api) for api in publications}) == 4
        _dispatch_updates(_update_snapshot()["update"], 0.1, 0.1)
        assert publications[-1].ticks == old_api.ticks == 1
        assert len(_update_snapshot()["update"]) == len(previous_updates) + 1
        assert sys.modules["Unrelated"] is unrelated_module
    finally:
        manager.shutdown()
        set_project_root(previous)
    assert _update_snapshot()["update"] == previous_updates


def test_package_preloads_with_matching_module_paths_are_isolated(tmp_path):
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))

    for reference, marker in (("vendor/first", "first"), ("vendor/second", "second")):
        source = _source(tmp_path / marker, reference)
        package_module = source / "editor" / "shared_name"
        package_module.mkdir(parents=True)
        (package_module / "__init__.py").write_text("", encoding="utf-8")
        (package_module / "service.py").write_text(f"VALUE = {marker!r}\n", encoding="utf-8")
        (package_module / "lifecycle.py").write_text(
            "from pathlib import Path\n"
            "from infernux.lifecycle import InxPreload\n"
            "from .service import VALUE\n"
            "class SharedNamePreload(InxPreload):\n"
            "    def preload(self, context):\n"
            "        Path(context.project_root, VALUE + '.txt').write_text(VALUE)\n",
            encoding="utf-8",
        )
        package = _export(source, tmp_path / f"{marker}.inxpkg")
        state = manager.install_package(str(package), install_dependencies=False)
        assert state.loaded

    assert (project / "first.txt").read_text(encoding="utf-8") == "first"
    assert (project / "second.txt").read_text(encoding="utf-8") == "second"
    module_names = {
        state.module_name for state in manager.preloads.states.values() if state.type_name == "SharedNamePreload"
    }
    assert len(module_names) == 2
    assert all(name.startswith("_infernux_packages.") for name in module_names)


def test_stubborn_plugin_does_not_block_unrelated_plugin_lifecycle(tmp_path, monkeypatch):
    stubborn = _source(tmp_path / "stubborn", "vendor/stubborn-slice")
    (stubborn / "runtime").mkdir()
    (stubborn / "runtime" / "service.py").write_text(
        "from infernux.lifecycle import InxPreload\n"
        "class Stubborn(InxPreload):\n"
        "    def preload(self, context): pass\n"
        "    def unload(self): raise RuntimeError('restart only me')\n",
        encoding="utf-8",
    )
    safe = _source(tmp_path / "safe", "vendor/safe-slice")
    (safe / "runtime").mkdir()
    (safe / "runtime" / "service.py").write_text(
        "from pathlib import Path\n"
        "from infernux.lifecycle import InxPreload\n"
        "class Safe(InxPreload):\n"
        "    def preload(self, context): self.root=context.project_root\n"
        "    def unload(self): Path(self.root, 'safe-unloaded.txt').write_text('yes')\n",
        encoding="utf-8",
    )
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))
    manager.install_package(
        str(_export(stubborn, tmp_path / "stubborn-slice.inxpkg")),
        install_dependencies=False,
    )
    manager.install_package(
        str(_export(safe, tmp_path / "safe-slice.inxpkg")),
        install_dependencies=False,
    )
    stubborn_state = next(
        state for state in manager.preloads.states.values() if state.package_reference == "vendor/stubborn-slice"
    )
    stubborn_instance = stubborn_state.instance
    monkeypatch.setattr(
        manager.preloads,
        "_source_paths",
        lambda: pytest.fail("targeted lifecycle performed a full AST scan"),
    )

    manager.set_enabled("vendor/safe-slice", False)
    assert stubborn_state.instance is stubborn_instance
    manager.set_enabled("vendor/safe-slice", True)
    manager.uninstall("vendor/safe-slice")
    assert stubborn_state.instance is stubborn_instance
    assert (project / "safe-unloaded.txt").is_file()

    stubborn_state.instance.unload = lambda: None
    manager.shutdown()


def test_native_guid_index_avoids_meta_walk_and_drives_preload_catalog(tmp_path, monkeypatch):
    project = _project(tmp_path / "project")
    script = project / "Assets" / "indexed.py"
    script.write_text(
        "from infernux.lifecycle import InxPreload\nclass Indexed(InxPreload):\n    def preload(self, context): pass\n",
        encoding="utf-8",
    )
    guid = "1234567890abcdef1234567890abcdef"
    subasset_guid = "abcdef1234567890abcdef1234567890"
    Path(str(script) + ".meta").write_text(_meta(guid), encoding="utf-8")

    class Database:
        project_root = str(project)

        @staticmethod
        def get_all_guids():
            return [guid, subasset_guid]

        @staticmethod
        def get_path_from_guid(value):
            return str(script) if value == guid else str(project / "Assets" / "model.glb") + "::subanim:Walk"

    class Engine:
        @staticmethod
        def get_asset_database():
            return Database()

    monkeypatch.setattr(
        "infernux.plugins.project_index._scan_guid_paths",
        lambda _root: pytest.fail("native GUID catalog fell back to os.walk"),
    )
    import os
    isfile = os.path.isfile

    def physical_asset(path):
        assert "::" not in str(path), "virtual subassets must not enter filesystem discovery"
        return isfile(path)

    monkeypatch.setattr(os.path, "isfile", physical_asset)
    paths, native = project_guid_paths(str(project), engine=Engine())
    assert native is True
    assert paths == {guid: str(script.resolve())}
    manager = PluginManager(str(project), engine=Engine())
    manager.reload_all()
    assert [state.type_name for state in manager.preloads.states.values()] == ["Indexed"]


def test_static_preload_discovery_imports_only_lifecycle_candidates(tmp_path):
    project = _project(tmp_path / "project")
    marker = project / "ordinary-imported.txt"
    (project / "Assets" / "ordinary.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('wrong')\n",
        encoding="utf-8",
    )
    lifecycle = project / "Assets" / "startup.py"
    lifecycle.write_text(
        "from abc import abstractmethod\n"
        "from pathlib import Path\n"
        "from infernux.lifecycle import InxPreload as Lifecycle\n"
        "class Foundation(Lifecycle):\n"
        "    @abstractmethod\n"
        "    def configure(self): ...\n"
        "    def preload(self, context): ...\n"
        "class Startup(Foundation):\n"
        "    def configure(self): return None\n"
        "    def preload(self, context):\n"
        "        Path(context.project_root, 'preloaded.txt').write_text(context.script_guid)\n"
        "    def unload(self):\n"
        "        Path(__file__).with_name('unloaded.txt').write_text('yes')\n",
        encoding="utf-8",
    )
    manager = PluginManager.startup(str(project))
    states = manager.preloads.snapshots()
    assert len(states) == 1
    assert states[0]["type_name"] == "Startup"
    assert states[0]["loaded"] is True
    assert not marker.exists()
    first_guid = (project / "preloaded.txt").read_text(encoding="utf-8")
    manager.reload_all()
    assert (project / "Assets" / "unloaded.txt").is_file()
    assert (project / "preloaded.txt").read_text(encoding="utf-8") == first_guid


def test_preload_hot_reload_keeps_public_namespace_identity_and_last_good_state(tmp_path):
    import infernux as public_namespace

    project = _project(tmp_path / "project")
    lifecycle = project / "Assets" / "public_api.py"

    def write_source(value: str) -> None:
        lifecycle.write_text(
            "import infernux as inx\n"
            "class PublicApi(inx.InxPreload):\n"
            "    def preload(self, context):\n"
            f"        inx._community_reload_probe = {value!r}\n"
            "        context.add_cleanup(lambda: delattr(inx, '_community_reload_probe'))\n",
            encoding="utf-8",
        )

    write_source("first")
    manager = PluginManager.startup(str(project))
    assert public_namespace._community_reload_probe == "first"
    assert sys.modules["infernux"] is public_namespace

    published = next(iter(manager.preloads.states.values())).instance
    stamp = lifecycle.stat()
    os.utime(lifecycle, ns=(stamp.st_atime_ns + 1_000_000_000, stamp.st_mtime_ns))
    assert manager.preloads.reload_path(str(lifecycle)) == ()
    assert next(iter(manager.preloads.states.values())).instance is published

    write_source("second")
    states = manager.preloads.reload_path(str(lifecycle))
    assert len(states) == 1 and states[0].loaded
    assert public_namespace._community_reload_probe == "second"
    assert sys.modules["infernux"] is public_namespace

    lifecycle.write_text("class PublicApi(\n", encoding="utf-8")
    retained = manager.preloads.reload_path(str(lifecycle))
    assert retained == states
    assert public_namespace._community_reload_probe == "second"
    assert sys.modules["infernux"] is public_namespace
    manager.shutdown()


def test_preload_partial_failure_runs_owned_cleanups_in_reverse_order(tmp_path):
    project = _project(tmp_path / "project")
    lifecycle = project / "Assets" / "partial_service.py"
    lifecycle.write_text(
        "from pathlib import Path\n"
        "from infernux.lifecycle import InxPreload\n"
        "class PartialService(InxPreload):\n"
        "    def preload(self, context):\n"
        "        output = Path(context.project_root, 'cleanup-order.txt')\n"
        "        def record(value):\n"
        "            output.write_text(output.read_text() + value if output.exists() else value)\n"
        "        context.add_cleanup(lambda: record('A'))\n"
        "        context.add_cleanup(lambda: record('B'))\n"
        "        raise RuntimeError('candidate failed after acquisition')\n",
        encoding="utf-8",
    )

    manager = PluginManager.startup(str(project))
    state = next(iter(manager.preloads.states.values()))
    assert state.loaded is False
    assert "candidate failed after acquisition" in state.error
    assert (project / "cleanup-order.txt").read_text(encoding="utf-8") == "BA"
    assert state.cleanup_callbacks == []
    manager.shutdown()


def test_preload_change_reloads_only_its_dependency_slice(tmp_path, monkeypatch):
    project = _project(tmp_path / "project")

    def write_preload(path: Path, label: str, revision: int) -> None:
        path.write_text(
            "from pathlib import Path\n"
            "from infernux.lifecycle import InxPreload\n"
            f"class {label}(InxPreload):\n"
            f"    revision = {revision}\n"
            "    def preload(self, context): pass\n"
            f"    def unload(self): Path(__file__).with_name('{label}.unloaded').write_text('yes')\n",
            encoding="utf-8",
        )

    first = project / "Assets" / "first.py"
    second = project / "Assets" / "second.py"
    write_preload(first, "First", 1)
    write_preload(second, "Second", 1)
    manager = PluginManager.startup(str(project))
    before = {state.type_name: state.instance for state in manager.preloads.states.values()}
    monkeypatch.setattr(
        manager.preloads,
        "_source_paths",
        lambda: pytest.fail("incremental reload performed a full source walk"),
    )

    write_preload(first, "First", 2)
    manager._on_script_catalog_changed(str(first), "modified")
    after = {state.type_name: state.instance for state in manager.preloads.states.values()}

    assert after["First"] is not before["First"]
    assert after["Second"] is before["Second"]
    assert (project / "Assets" / "First.unloaded").is_file()
    assert not (project / "Assets" / "Second.unloaded").exists()


def test_preload_cross_module_inheritance_move_disable_and_restart_diagnostic(tmp_path):
    source = _source(tmp_path / "source", "vendor/lifecycle")
    runtime = source / "runtime"
    runtime.mkdir()
    (runtime / "foundation.py").write_text(
        "from abc import abstractmethod\n"
        "from infernux.lifecycle import InxPreload\n"
        "class Foundation(InxPreload):\n"
        "    @abstractmethod\n"
        "    def configure(self): ...\n"
        "    def preload(self, context): ...\n",
        encoding="utf-8",
    )
    (runtime / "startup.py").write_text(
        "from pathlib import Path\n"
        "from .foundation import Foundation\n"
        "class Startup(Foundation):\n"
        "    def configure(self): return None\n"
        "    def preload(self, context):\n"
        "        Path(context.project_root, 'package-loaded.txt').write_text(context.script_guid)\n"
        "        context.require_restart('native test state')\n"
        "    def unload(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    package = _export(source, tmp_path / "lifecycle.inxpkg")
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))
    state = manager.install_package(str(package), install_dependencies=False)
    assert state.loaded is True
    assert state.restart_required is True
    assert state.lifecycle[0]["restart_reason"] == "native test state"
    original_identity = state.lifecycle[0]["identity"]

    old = project / "Packages/vendor/lifecycle/runtime/startup.py"
    moved = project / "Packages/vendor/lifecycle/runtime/boot.py"
    old.replace(moved)
    Path(str(old) + ".meta").replace(Path(str(moved) + ".meta"))
    state = manager.reload("vendor/lifecycle")
    assert state.lifecycle[0]["identity"] == original_identity

    disabled = manager.set_enabled("vendor/lifecycle", False)
    assert disabled.enabled is False
    assert disabled.loaded is False
    assert disabled.lifecycle == ()
    manager.set_enabled("vendor/lifecycle", True)
    assert (project / "package-loaded.txt").is_file()


def test_package_reload_preflights_every_source_before_unloading_live_state(tmp_path):
    source = _source(tmp_path / "source", "vendor/transactional-reload")
    runtime = source / "runtime"
    runtime.mkdir()
    (runtime / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (runtime / "startup.py").write_text(
        "from pathlib import Path\n"
        "from infernux.lifecycle import InxPreload\n"
        "class Startup(InxPreload):\n"
        "    def preload(self, context):\n"
        "        self.root = Path(context.project_root)\n"
        "    def unload(self):\n"
        "        (self.root / 'unexpected-unload.txt').write_text('unloaded')\n",
        encoding="utf-8",
    )
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))
    manager.install_package(
        str(_export(source, tmp_path / "transactional-reload.inxpkg")),
        install_dependencies=False,
    )
    before = tuple(manager.preloads.states.values())
    installed_helper = project / "Packages/vendor/transactional-reload/runtime/helper.py"
    installed_helper.write_text("def broken(:\n", encoding="utf-8")

    result = manager.preloads.reload_package("vendor/transactional-reload")

    assert result == before
    assert tuple(manager.preloads.states.values()) == before
    assert not (project / "unexpected-unload.txt").exists()
    assert "SyntaxError" in manager.preloads.failures[path_key(installed_helper)]
    manager.shutdown()


@pytest.mark.parametrize(
    "role,refresh",
    [("runtime", refresh) for refresh in ("reload_all", "reload_path", "reload_package")] + [("editor", "reload_all")],
)
def test_player_preload_refresh_respects_current_role_and_package_boundary(tmp_path, request, role, refresh):
    # A reference containing "editor" is not an Editor role directory.
    reference = "vendor/editor/tool"
    source = _source(tmp_path / "source", reference)
    (source / "runtime").mkdir()
    (source / "runtime" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    package = _export(source, tmp_path / "role.inxpkg")
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project), runtime=True)
    request.addfinalizer(manager.preloads.unload_all)
    manager.install_package(str(package), install_dependencies=False)
    script = project / "Packages" / reference / role / "added.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "from infernux.lifecycle import InxPreload\nclass Added(InxPreload):\n    def preload(self, context): pass\n",
        encoding="utf-8",
    )
    script.with_suffix(".py.meta").write_text(_meta("b" * 32), encoding="utf-8")
    if refresh == "reload_path":
        manager.preloads.reload_path(str(script))
    elif refresh == "reload_package":
        manager.preloads.reload_package(reference)
    else:
        manager.preloads.reload_all()
    assert not manager.preloads.failures
    assert len(manager.preloads.states) == (1 if role == "runtime" else 0)


def test_package_dependency_preload_and_reverse_unload_order(tmp_path):
    def lifecycle_source(reference: str, label: str) -> Path:
        source = _source(tmp_path / label, reference)
        runtime = source / "runtime"
        runtime.mkdir()
        (runtime / "startup.py").write_text(
            "from pathlib import Path\n"
            "from infernux.lifecycle import InxPreload\n"
            f"LABEL = {label!r}\n"
            "class Startup(InxPreload):\n"
            "    def preload(self, context):\n"
            "        path=Path(context.project_root, 'order.log')\n"
            "        with path.open('a', encoding='utf-8') as stream: stream.write('preload:' + LABEL + '\\n')\n"
            "    def unload(self):\n"
            "        path=Path(__file__).parents[4] / 'order.log'\n"
            "        with path.open('a', encoding='utf-8') as stream: stream.write('unload:' + LABEL + '\\n')\n",
            encoding="utf-8",
        )
        return source

    dependency = lifecycle_source("vendor/dependency", "dependency")
    parent = lifecycle_source("vendor/parent", "parent")
    (parent / "requirements.txt").write_text("vendor/dependency\n", encoding="utf-8")
    dependency_package = _export(dependency, tmp_path / "dependency.inxpkg")
    parent_package = _export(parent, tmp_path / "parent.inxpkg")
    project = _project(tmp_path / "project")
    registry = PluginRegistry(str(project))
    registry.add_package(
        "vendor/dependency",
        source={"type": "local", "location": str(dependency_package)},
    )
    manager = PluginManager(str(project))
    manager.install_package(str(parent_package))
    with pytest.raises(RuntimeError, match="required by enabled plugins"):
        manager.set_enabled("vendor/dependency", False)
    (project / "order.log").write_text("", encoding="utf-8")
    manager.reload_all()
    assert (project / "order.log").read_text(encoding="utf-8").splitlines() == [
        "unload:parent",
        "unload:dependency",
        "preload:dependency",
        "preload:parent",
    ]


@pytest.mark.parametrize("teardown", ["uninstall", "disable", "shutdown"])
def test_plugin_authoring_resources_follow_preload_lifetime(tmp_path, monkeypatch, teardown):
    from infernux.engine.interaction.commands import EditorCommandRegistry
    from infernux.engine.interaction.shortcuts import ShortcutRouter
    from infernux.engine.interaction.handles import EditorHandleRegistry

    monkeypatch.setattr(EditorCommandRegistry, "_instance", None)
    monkeypatch.setattr(ShortcutRouter, "_instance", None)
    monkeypatch.setattr(EditorHandleRegistry, "_instance", None)
    commands = EditorCommandRegistry.instance()
    shortcuts = ShortcutRouter.instance()
    handles = EditorHandleRegistry.instance()
    source = _source(tmp_path / "source", "vendor/authoring")
    editor = source / "editor"
    editor.mkdir()
    (editor / "startup.py").write_text(
        "from pathlib import Path\n"
        "from infernux.editor import register_handle_provider\n"
        "from infernux.lifecycle import InxPreload\n"
        "ROOT = ''\n"
        "def record(value):\n"
        "    with Path(ROOT, 'handles.log').open('a', encoding='utf-8') as stream: stream.write(str(value) + '\\n')\n"
        "def draw(handles):\n"
        "    handles.position('pivot', (0, 0, 0), lambda value: record(('change', tuple(value))), on_cancel=lambda value: record(('cancel', tuple(value))))\n"
        "from infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry\n"
        "from infernux.engine.interaction.shortcuts import KeyChord, ShortcutBinding, ShortcutRouter\n"
        "EditorCommandRegistry.instance().register(EditorCommand('vendor.create', lambda ctx: 'created', display_name='创建关卡'))\n"
        "class Startup(InxPreload):\n"
        "    def preload(self, context):\n"
        "        global ROOT\n"
        "        ROOT = context.project_root\n"
        "        register_handle_provider('vendor.handles', draw, on_retire=lambda: record(('retire',)))\n"
        "        ShortcutRouter.instance().register(ShortcutBinding('vendor.create', KeyChord.parse('Ctrl+F9'), binding_id='vendor.create.key'))\n",
        encoding="utf-8",
    )
    package = _export(source, tmp_path / "authoring.inxpkg")
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project))
    try:
        manager.install_package(str(package), install_dependencies=False)
        assert commands.get("vendor.create").display_name == "创建关卡"
        assert len(shortcuts.bindings) == 1
        for _ in range(2):
            manager.reload("vendor/authoring")
            assert commands.get("vendor.create") is not None
            assert len(shortcuts.bindings) == 1
            assert not manager.preloads.failures
        handles.collect(object())
        assert handles.begin_capture("vendor.handles:pivot", (0, 0, 5), (0, 0, -1))
        handles.update_capture((1, 0, 5), (0, 0, -1))

        if teardown == "uninstall":
            manager.uninstall("vendor/authoring")
        elif teardown == "disable":
            manager.set_enabled("vendor/authoring", False)
        else:
            manager.shutdown()
        assert commands.get("vendor.create") is None
        assert shortcuts.bindings == ()
        assert handles.active_capture_id == ""
        assert handles.frame_handles == ()
        assert handles.providers == ()
        assert (project / "handles.log").read_text(encoding="utf-8").splitlines()[-3:] == [
            "('change', (0.0, 0.0, 0.0))",
            "('cancel', (0.0, 0.0, 0.0))",
            "('retire',)",
        ]
        if teardown == "disable":
            manager.set_enabled("vendor/authoring", True)
            assert commands.get("vendor.create") is not None
            assert len(shortcuts.bindings) == 1
    finally:
        manager.shutdown()


def test_requirements_choose_official_inxpackage_before_pip(tmp_path, monkeypatch):
    dependency = _source(tmp_path / "dependency", "vendor/dependency", version="2.1.0")
    (dependency / "runtime").mkdir()
    (dependency / "runtime" / "api.py").write_text("VALUE = 1\n", encoding="utf-8")
    dep_package = _export(dependency, tmp_path / "dependency.inxpkg")
    parent = _source(tmp_path / "parent", "vendor/parent")
    (parent / "requirements.txt").write_text("vendor/dependency>=2\nthird-party-wheel==1.0\n", encoding="utf-8")
    (parent / "Data.bin").write_bytes(b"parent")
    parent_package = _export(parent, tmp_path / "parent.inxpkg")
    project = _project(tmp_path / "project")
    registry = PluginRegistry(str(project))
    registry.add_package(
        "vendor/dependency",
        version="2.1.0",
        source={"type": "local", "location": str(dep_package)},
    )
    manager = PluginManager(str(project))
    calls = []
    monkeypatch.setattr(manager, "_project_python_executable", lambda: "project-python")
    environment = {}

    def run(command, cwd=None):
        calls.append((command, cwd))
        if command[2:4] == ["pip", "install"]:
            environment["third-party-wheel"] = "1.0"
        return SimpleNamespace(
            stdout=json.dumps([{"name": name, "version": version} for name, version in environment.items()])
            if command[2:5] == ["pip", "list", "--format=json"]
            else ""
        )

    monkeypatch.setattr(manager, "_run_process", run)
    manager.install_package(str(parent_package))
    assert {item["reference"] for item in manager.registry.installed()} == {
        "vendor/dependency",
        "vendor/parent",
    }
    install_calls = [command for command, _cwd in calls if command[2:4] == ["pip", "install"]]
    assert len(install_calls) == 1
    assert install_calls[0][:4] == ["project-python", "-m", "pip", "install"]
