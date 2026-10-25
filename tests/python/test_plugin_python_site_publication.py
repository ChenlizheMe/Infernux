from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess
import threading
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from infernux.plugins import PluginManager
from infernux.host.commands import MainThreadCommandQueue


@pytest.fixture
def site_install(tmp_path, monkeypatch):
    site_root = tmp_path / "site-packages"
    site_root.mkdir()
    monkeypatch.syspath_prepend(str(site_root))
    counter = SimpleNamespace(new=0, unrelated=0, nested=0)
    monkeypatch.setitem(sys.modules, "site_publication_counter", counter)
    (site_root / "old.pth").write_text("import site_publication_counter; site_publication_counter.unrelated += 1\n")
    manager = PluginManager(str(tmp_path / "project"))
    monkeypatch.setattr(manager, "_project_python_executable", lambda: sys.executable)
    environment = {}
    monkeypatch.setattr(manager, "_python_environment_snapshot", lambda *_args: dict(environment))

    def install(_command, cwd=None):
        if "--dry-run" in _command:
            Path(_command[_command.index("--report") + 1]).write_text(json.dumps({"install": [{
                "metadata": {"name": "inx-site-test", "version": "1.0"},
                "download_info": {"url": "file:///fixture/fixture.whl"},
            }]}), encoding="utf-8")
            return SimpleNamespace(stdout="Would install fixture wheel")
        package = site_root / "new_site_path"
        package.mkdir(exist_ok=True)
        (package / "inx_site_published_probe.py").write_text("VALUE = 42\n")
        (package / "nested.pth").write_text("import site_publication_counter; site_publication_counter.nested += 1\n")
        (site_root / "new.pth").write_text("new_site_path\nimport site_publication_counter; site_publication_counter.new += 1\n")
        info = site_root / "inx_site_test-1.0.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: inx-site-test\nVersion: 1.0\n")
        (info / "RECORD").write_text("new.pth,,\nnew_site_path/nested.pth,,\nnew_site_path/inx_site_published_probe.py,,\ninx_site_test-1.0.dist-info/METADATA,,\n")
        environment['inx-site-test'] = '1.0'
        return SimpleNamespace(stdout="installed fixture wheel")

    monkeypatch.setattr(manager, "_run_process", install)
    yield manager, counter
    sys.modules.pop("inx_site_published_probe", None)


@pytest.mark.parametrize("previous_version", ["4.13.0", "4.14.0"])
def test_pip_stages_replaced_transitive_dependency_before_any_installation(site_install, monkeypatch, previous_version):
    importlib.import_module("anyio")
    from infernux_editor import read_pending

    manager, counter = site_install
    baseline = {"anyio": previous_version}
    registry = manager.registry.load()
    monkeypatch.setattr(manager, "_python_environment_snapshot", lambda *_: dict(baseline))
    monkeypatch.setattr(manager, "_resolve_python_install", lambda *_, **__: [{
        "metadata": {"name": "new-client", "version": "1.0"},
        "download_info": {"url": "https://example.test/new_client-1.0.whl"},
    }, {
        "metadata": {"name": "anyio", "version": "4.14.0"},
        "download_info": {"url": "https://example.test/anyio-4.14.0.whl"},
    }])
    monkeypatch.setattr(manager, "_run_process", lambda *_a, **_k: pytest.fail("live runtime must remain unchanged"))
    arguments = ["new-client==1.0"]
    if previous_version == "4.14.0":
        arguments.insert(0, "--force-reinstall")
    result = manager.install_pip(" ".join(arguments))
    assert result["restart_required"]
    assert manager.registry.load() == registry
    assert counter.new == counter.unrelated == 0
    pending = read_pending(manager.project_root)
    assert pending["before"] == baseline
    assert pending["request"]["arguments"] == arguments
    assert not pending["prepared"]
    manager.cancel_pending_install()
    assert read_pending(manager.project_root) is None


def test_unimported_dependency_updates_without_restart(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    module = site / "inx_unimported_dependency.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    info = site / "inx_unimported_dependency-1.0.dist-info"
    info.mkdir()
    metadata = info / "METADATA"
    metadata.write_text("Metadata-Version: 2.1\nName: inx-unimported-dependency\nVersion: 1.0\n", encoding="utf-8")
    (info / "RECORD").write_text("inx_unimported_dependency.py,,\ninx_unimported_dependency-1.0.dist-info/METADATA,,\n", encoding="utf-8")
    manager = PluginManager(str(tmp_path / "project"))
    monkeypatch.setattr(manager, "_project_python_executable", lambda: sys.executable)
    monkeypatch.setattr(manager, "_resolve_python_install", lambda *_, **__: [{
        "metadata": {"name": "inx-unimported-dependency", "version": "2.0"},
        "download_info": {"url": "https://example.test/dependency-2.0.whl"},
    }])

    def install(*_a, **_k):
        assert "inx_unimported_dependency" not in sys.modules
        module.write_text("VALUE = 2\n", encoding="utf-8")
        metadata.write_text("Metadata-Version: 2.1\nName: inx-unimported-dependency\nVersion: 2.0\n", encoding="utf-8")
        return SimpleNamespace(stdout="installed")

    monkeypatch.setattr(manager, "_run_process", install)
    try:
        result = manager.install_pip("inx-unimported-dependency==2.0")
        assert not result.get("restart_required") and manager.pending_install is None
        assert importlib.import_module("inx_unimported_dependency").VALUE == 2
    finally:
        sys.modules.pop("inx_unimported_dependency", None)


def test_wheel_requirements_use_the_wheels_exact_version():
    from infernux.plugins.manager import _pip_requirement_targets, _requirements_satisfied

    for wheel in ("C:/packages/anyio-4.14.0-py3-none-any.whl", "https://example.test/anyio-4.14.0-py3-none-any.whl"):
        targets = _pip_requirement_targets([wheel])
        assert targets == ({"name": "anyio", "requirement": "anyio==4.14.0"},)
        assert _requirements_satisfied([targets[0]["requirement"]], {"anyio": "4.14.0"})
        assert not _requirements_satisfied([targets[0]["requirement"]], {"anyio": "4.13.0"})


def test_preparation_retries_publication_without_replacing_dependencies_again(tmp_path):
    import infernux_editor

    script = '''
import os, sys
sys.path.insert(0, sys.argv[1])
import infernux_editor as editor
import infernux_project_lock as locks
project = sys.argv[2]
token = "preparation-test"
locks.reserve(project, os.getpid(), token, "editor", "preparing")
os.environ["_INFERNUX_PROJECT_LOCK_TOKEN"] = token
before = {"anyio": "4.13.0"}
editor.save_pending(project, {"before": before, "request": {"kind": "pip"}, "install": [{
    "metadata": {"name": "anyio", "version": "4.14.0"},
    "download_info": {"url": "https://example.test/anyio-4.14.0.whl"},
}]})
calls = []
def run(project, arguments):
    calls.append(arguments)
    if "--publish" in arguments:
        raise RuntimeError("publication failed")
    return "installed"
editor._run = run
try:
    editor.prepare(project)
except RuntimeError as error:
    assert str(error) == "publication failed"
else:
    raise AssertionError("publication error was swallowed")
pending = editor.read_pending(project)
assert pending["prepared"] and pending["before"] == before
assert len(calls) == 2 and "--no-deps" in calls[0]
def publish(project, arguments):
    calls.append(arguments)
    assert "--publish" in arguments
    return "published"
editor._run = publish
editor.prepare(project)
assert len(calls) == 3 and editor.read_pending(project) is None
assert "infernux" not in sys.modules and "anyio" not in sys.modules
locks.remove_lock(locks.lock_path(project), token)
'''
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(infernux_editor.__file__).parent), str(tmp_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_restart_waits_until_the_previous_process_has_exited():
    from infernux_editor import _wait_for_process

    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline()"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    completed = threading.Event()
    errors = []

    def wait():
        try:
            _wait_for_process(child.pid)
        except Exception as error:
            errors.append(error)
        finally:
            completed.set()

    worker = threading.Thread(target=wait)
    worker.start()
    try:
        assert not completed.wait(0.05), (child.poll(), errors)
        child.communicate(b"exit\n", timeout=10)
        assert completed.wait(5) and not errors
    finally:
        if child.poll() is None:
            child.kill()
        worker.join(5)


@pytest.mark.parametrize("entrypoint", ["requirements", "pip"])
def test_new_dependency_site_hooks_are_published_in_the_same_process(site_install, entrypoint):
    manager, counter = site_install
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("inx_site_published_probe")
    if entrypoint == "requirements":
        manager._install_pip_lines(["inx-site-test==1.0\n"])
    else:
        manager.install_pip("inx-site-test==1.0")
    assert importlib.import_module("inx_site_published_probe").VALUE == 42
    assert counter.new == 1
    assert counter.unrelated == counter.nested == 0
    # No-op installation must not re-execute any site's startup hooks.
    manager._install_pip_lines(["inx-site-test==1.0\n"])
    assert counter.new == 1


def test_installing_for_another_interpreter_does_not_mutate_live_paths(site_install, monkeypatch):
    manager, counter = site_install
    monkeypatch.setattr(manager, "_project_python_executable", lambda: "another-python")
    paths = list(sys.path)
    manager._install_pip_lines(["inx-site-test==1.0\n"])
    assert sys.path == paths
    assert counter.new == 0


def test_startup_requirement_plan_is_prepared_before_loading_the_editor(site_install, monkeypatch):
    from infernux_editor import _write_pending

    manager, counter = site_install
    monkeypatch.setattr(manager.registry, "installed", lambda: [{
        "reference": "vendor/site-test", "enabled": True,
        "python_requirements": [{"requirement": "inx-site-test==1.0"}],
    }])
    reconciled = []
    monkeypatch.setattr(manager.registry, "record_python_reconciliation", lambda **value: reconciled.append(value))
    with pytest.raises(RuntimeError, match="prepare"):
        manager._reconcile_python_requirements_for_startup()
    operation = manager._plan_startup_python()
    assert operation["request"]["requirements"] == ["inx-site-test==1.0"]
    assert counter.new == 0 and not reconciled
    manager._run_process([sys.executable, "-m", "pip", "install", "fixture.whl"])
    _write_pending(manager.project_root, {**operation, "prepared": True})
    manager.installation_only = True
    manager._complete_pending_install()
    assert manager._reconcile_python_requirements_for_startup() == ()
    assert counter.new == 0
    assert len(reconciled) == 1


def _dependency_wheel(directory, name, version):
    directory.mkdir(parents=True, exist_ok=True)
    module = name.replace("-", "_")
    wheel = directory / f"{module}-{version}-py3-none-any.whl"
    info = f"{module}-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{module}.py", f"VERSION = {version!r}\n")
        archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    return wheel


@pytest.mark.parametrize("entrypoint", ["defer", "release"])
def test_uninstall_resolver_releases_its_owner_before_restoring_baseline(tmp_path, monkeypatch, entrypoint):
    name = "inx-uninstall-baseline-test"
    wheel = _dependency_wheel(tmp_path / "wheels", name, "1.0")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheel.parent))
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    manager = PluginManager(str(tmp_path / "project"))
    manager.registry.record_install(
        {"reference": "vendor/baseline", "version": "1.0"}, files=(),
        control={"guid": "62a3482c971c469485877f2e5b41a2b5", "path_hint": "Packages/vendor/baseline/inx_package.json"},
        python_requirements=[{"name": name, "requirement": f"{name}==2.0"}],
        python_environment={name: "2.0"},
        python_changes=[{"name": name, "before": "1.0", "after": "2.0"}],
    )
    monkeypatch.setattr(manager, "_python_environment_snapshot", lambda *_: {name: "2.0"})
    monkeypatch.setattr(manager, "_python_dependencies_in_use", lambda _: ())
    monkeypatch.setattr(manager, "_python_change_scope", lambda _: nullcontext(()))
    run = manager._run_process
    installations = []

    def resolve_or_record(command, **kwargs):
        if "--dry-run" in command:
            return run(command, **kwargs)
        installations.append(command)
        return SimpleNamespace(stdout="installation is not needed for the resolver regression")

    monkeypatch.setattr(manager, "_run_process", resolve_or_record)
    if entrypoint == "defer":
        assert not manager._defer_uninstall_python("vendor/baseline")
        assert installations == []
    else:
        assert manager._release_python_dependencies("vendor/baseline") == {name: "2.0"}
        assert len(installations) == 1 and wheel.as_uri() in installations[0]


@pytest.mark.parametrize("owner", ["@project", "vendor/source"])
@pytest.mark.parametrize("target_version", ["1.0", "2.0"])
def test_startup_direct_url_uses_source_before_registry_pin(tmp_path, monkeypatch, owner, target_version):
    from infernux_editor import _write_pending

    name = "inx-direct-source-test"
    wheel = _dependency_wheel(tmp_path / "new-source", name, target_version)
    requirement = f"{name} @ {wheel.as_uri()}"
    site = tmp_path / "site"
    info = site / "inx_direct_source_test-1.0.dist-info"
    info.mkdir(parents=True)
    metadata = info / "METADATA"
    metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8")
    direct_url = info / "direct_url.json"
    direct_url.write_text(json.dumps({"url": (tmp_path / "old-source.whl").as_uri()}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    manager = PluginManager(str(tmp_path / "project"))
    if owner == "@project":
        manager.registry.record_python_install(
            syntax=f"{name}==1.0", command=(), output="", owner=owner,
            requirements=[f"{name}==1.0"],
            dependency_requirements=[{"name": name, "requirement": f"{name}==1.0"}],
            python_environment={name: "1.0"},
        )
        authored = Path(manager.project_root) / "ProjectSettings" / "requirements.txt"
        authored.write_text(requirement + "\n", encoding="utf-8")
    else:
        manager.registry.record_install(
            {"reference": owner, "version": "1.0"}, files=(),
            control={"guid": "62a3482c971c469485877f2e5b41a2b5", "path_hint": f"Packages/{owner}/inx_package.json"},
            python_requirements=[{"name": name, "requirement": requirement}],
            python_environment={name: "1.0"},
        )
    environment = {name: "1.0"}
    monkeypatch.setattr(manager, "_python_environment_snapshot", lambda *_: dict(environment))
    with pytest.raises(RuntimeError, match="prepare"):
        manager._reconcile_python_requirements_for_startup()
    operation = manager._plan_startup_python()
    assert operation["request"]["requirements"] == [requirement]
    assert operation["install"][0]["metadata"] == {"name": name, "version": target_version}
    _write_pending(manager.project_root, {**operation, "prepared": True})
    manager.installation_only = True
    with pytest.raises(RuntimeError, match="Prepared startup requirements"):
        manager._complete_pending_install()
    direct_url.write_text(json.dumps({"url": wheel.as_uri()}), encoding="utf-8")
    metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {target_version}\n", encoding="utf-8")
    environment[name] = target_version
    manager._complete_pending_install()
    assert manager._reconcile_python_requirements_for_startup() == ()
    assert manager._plan_startup_python() is None


def test_failed_pip_transaction_does_not_publish_site_hooks(site_install, monkeypatch):
    manager, counter = site_install
    install = manager._run_process

    def fail(*args, **kwargs):
        install(*args, **kwargs)
        raise RuntimeError("pip failed")

    monkeypatch.setattr(manager, "_run_process", fail)
    monkeypatch.setattr(manager, "_restore_python_environment", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="pip failed"):
        manager._install_pip_lines(["inx-site-test==1.0\n"])
    assert counter.new == 0


def test_background_dependency_install_publishes_on_editor_owner_thread(site_install, monkeypatch):
    manager, counter = site_install
    manager.engine = object()
    queue = MainThreadCommandQueue()
    monkeypatch.setattr(MainThreadCommandQueue, "_instance", queue)
    queue.drain()
    calls = []
    import site
    original = site.addpackage

    def addpackage(*args):
        calls.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(site, "addpackage", addpackage)
    errors = []
    done = threading.Event()

    def install():
        try:
            manager._install_pip_lines(["inx-site-test==1.0\n"])
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=install)
    worker.start()
    try:
        while not done.wait(0.01):
            queue.drain()
        worker.join(1)
        assert not errors
        assert calls == [threading.get_ident()]
        assert counter.new == 1
    finally:
        queue.cancel_pending()
        worker.join(2)


def test_install_owner_shutdown_cancels_site_request_before_join_and_registry_publication(site_install, monkeypatch):
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.ui.plugin_install_progress import PluginInstallProgressService

    manager, counter = site_install
    manager.engine = object()
    core = EditorInteractionCore()
    service = PluginInstallProgressService()
    monkeypatch.setattr(PluginInstallProgressService, "_instance", service)
    queue = MainThreadCommandQueue()
    monkeypatch.setattr(MainThreadCommandQueue, "_instance", queue)
    queue.drain()
    pending = threading.Event()
    submit = queue.submit

    def observe(name, fn, **kwargs):
        future = submit(name, fn, **kwargs)
        if name == "plugin.python-site-publish":
            pending.set()
        return future

    monkeypatch.setattr(queue, "submit", observe)
    baseline = manager.registry.load()
    restored = []
    monkeypatch.setattr(manager, "_restore_python_environment", lambda before, **kwargs: restored.append(before))
    assert service.begin(label="pip", work=lambda report: manager.install_pip("inx-site-test==1.0", progress=report),
                         complete=lambda *args: pytest.fail("retired callback ran"))
    transaction = service._transaction
    transaction.presented_phase = "opening"
    service.post_present_tick()
    try:
        assert pending.wait(3)
        service.shutdown(modals=core.modals)
        assert not transaction.worker.is_alive()
        assert isinstance(transaction.worker_error, TimeoutError)
        assert "Host service stopped" in str(transaction.worker_error)
        queue.drain()
        assert counter.new == 0
        assert restored == [{}]
        assert manager.registry.load() == baseline
        assert not service.is_active
    finally:
        transaction.commands.close()
        queue.cancel_pending()
        transaction.worker.join(3)
        core.shutdown()
