"""Real package publication must validate the proposed dependency graph first."""
import json
from pathlib import Path

import pytest

from infernux.engine import player_package_native
from infernux.plugins import InxPackage, PluginManager
from infernux.version import ENGINE_VERSION


def make_package(source, reference, *, version="1.0.0", requirements=""):
    source.mkdir(parents=True, exist_ok=True)
    (source / "inx_package.json").write_text(json.dumps({
        "reference": reference, "version": version, "engine": "==" + ENGINE_VERSION,
    }), encoding="utf-8")
    (source / "requirements.txt").write_text(requirements, encoding="utf-8")
    runtime = source / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "startup.py").write_text(
        "from pathlib import Path\nfrom infernux.lifecycle import InxPreload\n"
        "class Startup(InxPreload):\n"
        "    def preload(self, context):\n"
        "        self.events = Path(context.project_root) / 'events.txt'\n"
        f"        with self.events.open('a') as f: f.write('load:{reference}:{version}\\n')\n"
        "    def unload(self):\n"
        f"        with self.events.open('a') as f: f.write('unload:{reference}:{version}\\n')\n",
        encoding="utf-8",
    )
    archive = source.parent / (source.name + "-" + version + ".inxpkg")
    return InxPackage.export_source(str(source), str(archive)).package_path


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    assert not player_package_native.using_test_backend()
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(tmp_path / "shared"))
    root = tmp_path / "project"
    (root / "Assets").mkdir(parents=True)
    (root / "ProjectSettings").mkdir()
    previous = PluginManager._instance
    manager = PluginManager(str(root), runtime=True)
    try:
        yield manager, root
    finally:
        manager.shutdown()
        PluginManager._instance = previous


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and (path.parts[len(root.parts)] in {"Packages", "ProjectSettings"}
                                   or path.name == "events.txt")}


@pytest.mark.parametrize("syntax", ("reference", "inx", "nested"))
@pytest.mark.parametrize("install_dependencies", (True, False))
def test_disabled_dependency_rejects_parent_before_lifecycle_or_publication(
    plugins, tmp_path, syntax, install_dependencies,
):
    manager, root = plugins
    child = make_package(tmp_path / "child", "test/child")
    assert manager.install_package(child).loaded
    manager.set_enabled("test/child", False)
    parent_source = tmp_path / "parent"
    if syntax == "nested":
        parent_source.mkdir()
        (parent_source / "child.inxpkg").write_bytes(Path(child).read_bytes())
    requirement = {"reference": "test/child", "inx": "inx:test/child", "nested": "child.inxpkg"}[syntax]
    parent = make_package(parent_source, "test/parent", requirements=requirement + "\n")
    before = snapshot(root)
    with pytest.raises(RuntimeError, match="requires enabled plugins"):
        manager.install_package(parent, install_dependencies=install_dependencies)
    assert snapshot(root) == before
    assert manager.registry.installed_record("test/parent") is None
    assert not manager.registry.installed_record("test/child")["enabled"]


@pytest.mark.parametrize("depth", (0, 1, 3))
@pytest.mark.parametrize("install_dependencies", (True, False))
def test_cycle_update_is_rejected_before_stopping_old_plugin(plugins, tmp_path, depth, install_dependencies):
    manager, root = plugins
    child_source = tmp_path / "child"
    assert manager.install_package(make_package(child_source, "test/child")).loaded
    previous = "test/child"
    for index in range(depth):
        reference = f"test/parent{index}"
        assert manager.install_package(make_package(
            tmp_path / f"parent{index}", reference, requirements=previous + "\n",
        )).loaded
        previous = reference
    before = snapshot(root)
    record = manager.registry.installed_record("test/child")
    incoming = make_package(child_source, "test/child", version="2.0.0", requirements=previous + "\n")
    with pytest.raises(RuntimeError, match="Circular"):
        manager.install_package(incoming, update=True, install_dependencies=install_dependencies)
    assert snapshot(root) == before
    assert manager.registry.installed_record("test/child") == record
    assert manager.states["test/child"].loaded
    manager.reload_all()
    assert all(state.loaded for state in manager.states.values())


def test_enabled_dependency_is_reused_without_restarting_its_service(plugins, tmp_path):
    manager, root = plugins
    assert manager.install_package(make_package(tmp_path / "child", "test/child")).loaded
    before = (root / "events.txt").read_text()
    parent = make_package(tmp_path / "parent", "test/parent", requirements="test/child\n")
    assert manager.install_package(parent).loaded
    assert (root / "events.txt").read_text() == before + "load:test/parent:1.0.0\n"


def test_disabled_parent_update_can_retain_disabled_dependency(plugins, tmp_path):
    manager, root = plugins
    assert manager.install_package(make_package(tmp_path / "child", "test/child")).loaded
    source = tmp_path / "parent"
    assert manager.install_package(make_package(source, "test/parent", requirements="test/child\n")).loaded
    manager.set_enabled("test/parent", False)
    manager.set_enabled("test/child", False)
    updated = make_package(source, "test/parent", version="2.0.0", requirements="test/child\n")
    state = manager.install_package(updated, update=True)
    assert not state.enabled and not state.loaded
    with pytest.raises(RuntimeError, match="requires enabled plugins"):
        manager.set_enabled("test/parent", True)
    manager.set_enabled("test/child", True)
    assert manager.set_enabled("test/parent", True).loaded


@pytest.mark.parametrize("prefix", ("", "inx:"))
def test_first_install_cannot_depend_on_itself(plugins, tmp_path, prefix, monkeypatch):
    manager, root = plugins
    archive = make_package(tmp_path / "self", "test/self", requirements=prefix + "test/self\n")
    monkeypatch.setattr(manager, "_install_pip_lines", lambda *_: pytest.fail("not a pip requirement"))
    before = snapshot(root)
    with pytest.raises(RuntimeError, match="Circular"):
        manager.install_package(archive)
    assert snapshot(root) == before


def test_cycle_through_new_dependency_preserves_old_live_graph(plugins, tmp_path):
    manager, root = plugins
    child_source = tmp_path / "child"
    assert manager.install_package(make_package(child_source, "test/child")).loaded
    parent = make_package(tmp_path / "parent", "test/parent", requirements="test/child\n")
    assert manager.install_package(parent).loaded
    helper = make_package(tmp_path / "helper", "test/helper", requirements="test/parent\n")
    manager.registry.add_package("test/helper", version="1.0.0", source={"type": "local", "location": helper})
    before = snapshot(root)
    incoming = make_package(child_source, "test/child", version="2.0.0", requirements="test/helper\n")
    with pytest.raises(RuntimeError, match="Circular"):
        manager.install_package(incoming, update=True)
    assert snapshot(root) == before
    assert manager.registry.installed_record("test/helper") is None
    assert all(state.loaded for state in manager.states.values())


@pytest.mark.parametrize("install_dependencies", (True, False))
def test_cold_manager_initializes_existing_dependency_once_in_graph_order(plugins, tmp_path, install_dependencies):
    manager, root = plugins
    assert manager.install_package(make_package(tmp_path / "child", "test/child")).loaded
    manager.shutdown()
    before = (root / "events.txt").read_text()
    cold = PluginManager(str(root), runtime=True)
    try:
        parent = make_package(tmp_path / "parent", "test/parent", requirements="test/child\n")
        assert cold.install_package(parent, install_dependencies=install_dependencies).loaded
        assert (root / "events.txt").read_text() == before + "load:test/child:1.0.0\nload:test/parent:1.0.0\n"
    finally:
        cold.shutdown()


def test_invalid_graph_is_rejected_before_python_dependency_install(plugins, tmp_path, monkeypatch):
    manager, root = plugins
    assert manager.install_package(make_package(tmp_path / "child", "test/child")).loaded
    manager.set_enabled("test/child", False)
    parent = make_package(tmp_path / "parent", "test/parent", requirements="test/child\nexample-python-dep==1\n")
    monkeypatch.setattr(manager, "_install_pip_lines", lambda *_: pytest.fail("graph must precede pip"))
    before = snapshot(root)
    with pytest.raises(RuntimeError, match="requires enabled plugins"):
        manager.install_package(parent)
    assert snapshot(root) == before


def test_skip_installer_records_dependency_and_enforces_later_disable_gate(plugins, tmp_path):
    manager, root = plugins
    assert manager.install_package(make_package(tmp_path / "child", "test/child")).loaded
    parent = make_package(tmp_path / "parent", "test/parent", requirements="test/child\n")
    assert manager.install_package(parent, install_dependencies=False).loaded
    assert manager.registry.installed_record("test/parent")["dependencies"] == ["test/child"]
    with pytest.raises(RuntimeError, match="required by enabled plugins"):
        manager.set_enabled("test/child", False)
