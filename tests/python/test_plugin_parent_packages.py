"""Isolated plugin imports execute authored parents using normal package semantics."""
from pathlib import Path
import py_compile
import sys

import pytest

from infernux.plugins.preload import _load_module, _module_name, _package_module_names
from test_plugin_preload_failures import package_project


STARTUP = '''from infernux.lifecycle import InxPreload
from . import VALUE
class Startup(InxPreload):
    def preload(self, context):
        self.value = VALUE
'''


def _instances(manager, state):
    return [manager.preloads.states[item['identity']].instance for item in state.lifecycle]


@pytest.mark.parametrize('value_location', ['parent', 'sibling'])
def test_native_package_import_resolves_parent_exports(package_project, value_location):
    _root, manager, install = package_project
    state = install('test/parent-exports', {
        'runtime/nested/__init__.py': "VALUE = 'ready'\n" if value_location == 'parent' else '',
        'runtime/nested/values.py': "VALUE = 'ready'\n",
        'runtime/nested/startup.py': STARTUP if value_location == 'parent' else STARTUP.replace('from . import VALUE', 'from .values import VALUE'),
    })
    assert state.loaded, state.error
    assert [instance.value for instance in _instances(manager, state)] == ['ready']


def test_parent_chain_executes_once_and_same_size_reload_uses_new_source(package_project):
    root, manager, install = package_project
    parent_source = (
        "from pathlib import Path\nfrom .helper import SUFFIX\n"
        "with Path(__file__).with_name('initializations.txt').open('a') as output:\n"
        "    output.write('init\\n')\n"
        "VALUE = 'one' + SUFFIX\n"
    )
    state = install('test/parent-chain', {
        'runtime/__init__.py': "ROOT_VALUE = 'root'\n",
        'runtime/nested/__init__.py': parent_source,
        'runtime/nested/helper.py': "from .. import ROOT_VALUE\nSUFFIX = '-' + ROOT_VALUE\n",
        'runtime/nested/a.py': STARTUP,
        'runtime/nested/b.py': STARTUP,
    })
    assert state.loaded, state.error
    assert [value.value for value in _instances(manager, state)] == ['one-root', 'one-root']
    directory = root / 'Packages/test/parent-chain/runtime/nested'
    assert (directory / 'initializations.txt').read_text().splitlines() == ['init']
    (directory / '__init__.py').write_text(parent_source.replace("'one'", "'two'"), encoding='utf-8')
    state = manager.reload('test/parent-chain')
    assert state.loaded, state.error
    assert [value.value for value in _instances(manager, state)] == ['two-root', 'two-root']
    assert (directory / 'initializations.txt').read_text().splitlines() == ['init', 'init']
    parent = sys.modules[_module_name(str(directory / '__init__.py'), str(root))]
    assert parent.a is sys.modules[parent.__name__ + '.a']
    assert parent.b is sys.modules[parent.__name__ + '.b']


def test_parent_importing_candidate_does_not_execute_candidate_twice(package_project):
    root, manager, install = package_project
    startup = (
        "from pathlib import Path\n"
        "with Path(__file__).with_name('executions.txt').open('a') as output:\n"
        "    output.write('module\\n')\n" + STARTUP
    )
    state = install('test/parent-imports-child', {
        'runtime/nested/__init__.py': "VALUE = 'ready'\nfrom . import startup\n",
        'runtime/nested/startup.py': startup,
    })
    assert state.loaded, state.error
    directory = root / 'Packages/test/parent-imports-child/runtime/nested'
    assert (directory / 'executions.txt').read_text().splitlines() == ['module']
    state = manager.reload('test/parent-imports-child')
    assert state.loaded, state.error
    assert (directory / 'executions.txt').read_text().splitlines() == ['module', 'module']


def test_parent_failure_retires_new_modules_and_owned_editor_contributions(package_project, monkeypatch):
    from infernux.engine.interaction.commands import EditorCommandRegistry

    root, manager, install = package_project
    monkeypatch.setattr(EditorCommandRegistry, '_instance', None)
    registry = EditorCommandRegistry.instance()
    state = install('test/parent-failure', {
        'editor/nested/__init__.py': (
            'from . import helper\n'
            'from infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry\n'
            "EditorCommandRegistry.instance().register(EditorCommand('parent.temporary', lambda ctx: None))\n"
            "raise RuntimeError('authored-parent-failed')\n"
        ),
        'editor/nested/helper.py': 'VALUE = 1\n',
        'editor/nested/startup.py': STARTUP,
    })
    assert not state.loaded and 'authored-parent-failed' in state.error
    assert registry.get('parent.temporary') is None
    assert _package_module_names(str(root), 'test/parent-failure') == ()
    directory = root / 'Packages/test/parent-failure/editor/nested'
    (directory / '__init__.py').write_text("from .helper import VALUE\n", encoding='utf-8')
    state = manager.reload('test/parent-failure')
    assert state.loaded and not state.error
    assert _instances(manager, state)[0].value == 1


@pytest.mark.parametrize('compiled', [False, True])
def test_parent_source_or_bytecode_import_has_correct_package_identity(tmp_path, compiled):
    root = tmp_path / 'Project'
    directory = root / 'Packages/test/compiled_parent/runtime/nested'
    directory.mkdir(parents=True)
    (directory / '__init__.py').write_text("VALUE = 'ready'\n", encoding='utf-8')
    source = directory / 'startup.py'
    source.write_text('from . import VALUE\nRESULT = VALUE\n', encoding='utf-8')
    if compiled:
        for path in (directory / '__init__.py', source):
            py_compile.compile(str(path), cfile=str(path) + 'c', doraise=True)
            path.unlink()
        source = source.with_suffix('.pyc')
    name = '_infernux_packages.test.compiled_parent.runtime.nested.startup'
    before = set(sys.modules)
    try:
        module = _load_module(name, str(source), str(root), 'test/compiled_parent')
        assert module.RESULT == 'ready'
        parent = sys.modules[name.rsplit('.', 1)[0]]
        assert parent.startup is module
        assert parent.__spec__.submodule_search_locations == [str(directory)]
        assert parent.__spec__.loader is not None
        assert _load_module(name, str(source), str(root), 'test/compiled_parent') is module
    finally:
        for key in sorted(set(sys.modules) - before, reverse=True):
            if key.startswith('_infernux_packages'):
                sys.modules.pop(key, None)


def test_failed_candidate_does_not_leave_new_helper_modules(package_project):
    root, manager, install = package_project
    state = install('test/candidate-failure', {
        'runtime/nested/__init__.py': '',
        'runtime/nested/helper.py': 'VALUE = 1\n',
        'runtime/nested/startup.py': 'from . import helper\nraise RuntimeError("candidate-import-failed")\n' + STARTUP,
    })
    assert not state.loaded and 'candidate-import-failed' in state.error
    assert _package_module_names(str(root), 'test/candidate-failure') == ()


@pytest.mark.parametrize('concrete_child', [False, True])
def test_abstract_parent_import_runs_once_and_keeps_cleanup_ownership(package_project, monkeypatch, concrete_child):
    from infernux.engine.interaction.commands import EditorCommandRegistry

    root, manager, install = package_project
    monkeypatch.setattr(EditorCommandRegistry, '_instance', None)
    commands = EditorCommandRegistry.instance()
    sources = {
        'editor/nested/__init__.py': (
            'from abc import abstractmethod\nfrom pathlib import Path\n'
            'from infernux.lifecycle import InxPreload\n'
            'from infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry\n'
            "EditorCommandRegistry.instance().register(EditorCommand('parent.abstract', lambda ctx: None))\n"
            "with Path(__file__).with_name('executions.txt').open('a') as output:\n"
            "    output.write('parent\\n')\n"
            'class Foundation(InxPreload):\n'
            '    @abstractmethod\n    def configure(self): ...\n'
            '    def preload(self, context): self.configure()\n'
        ),
    }
    if concrete_child:
        sources['editor/nested/startup.py'] = (
            'from . import Foundation\nclass Startup(Foundation):\n'
            '    def configure(self): self.ready = True\n'
        )
    state = install('test/abstract-parent', sources)
    assert state.loaded, state.error
    assert len(state.lifecycle) == int(concrete_child)
    counter = root / 'Packages/test/abstract-parent/editor/nested/executions.txt'
    assert counter.read_text().splitlines() == ['parent']
    assert commands.get('parent.abstract') is not None
    state = manager.reload('test/abstract-parent')
    assert state.loaded, state.error
    assert counter.read_text().splitlines() == ['parent', 'parent']
    manager.shutdown()
    assert commands.get('parent.abstract') is None
    assert _package_module_names(str(root), 'test/abstract-parent') == ()


def test_failed_new_child_preserves_the_already_loaded_parent_and_service(package_project):
    from infernux.plugins.preload import _ensure_script_guid

    root, manager, install = package_project
    state = install('test/live-parent', {
        'runtime/nested/__init__.py': "VALUE = 'ready'\n",
        'runtime/nested/good.py': STARTUP,
    })
    assert state.loaded, state.error
    original = _instances(manager, state)[0]
    directory = root / 'Packages/test/live-parent/runtime/nested'
    parent_name = _module_name(str(directory / '__init__.py'), str(root))
    original_parent = sys.modules[parent_name]
    helper = directory / 'new_helper.py'
    helper.write_text('VALUE = 2\n', encoding='utf-8')
    bad = directory / 'bad.py'
    bad.write_text('from . import new_helper\nraise RuntimeError("new-child-failed")\n' + STARTUP, encoding='utf-8')
    for path in (helper, bad):
        _ensure_script_guid(str(path), str(root))
    assert manager.catch_up_preloads() == ()
    state = manager.states['test/live-parent']
    assert not state.loaded and 'new-child-failed' in state.error
    assert _instances(manager, state)[0] is original
    assert sys.modules[parent_name] is original_parent
    assert original_parent.good is sys.modules[parent_name + '.good']
    for name in ('bad', 'new_helper'):
        assert parent_name + '.' + name not in sys.modules
        assert not hasattr(original_parent, name)


def test_module_only_cleanup_refusal_keeps_ownership_until_explicit_retry(package_project, monkeypatch):
    from infernux.plugins import preload
    from infernux.engine.interaction.commands import EditorCommandRegistry

    root, manager, install = package_project
    monkeypatch.setattr(EditorCommandRegistry, '_instance', None)
    commands = EditorCommandRegistry.instance()
    state = install('test/retained-parent', {'editor/nested/__init__.py': (
        'from abc import abstractmethod\nfrom infernux.lifecycle import InxPreload\n'
        'from infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry\n'
        "EditorCommandRegistry.instance().register(EditorCommand('parent.retained', lambda ctx: None))\n"
        'class Base(InxPreload):\n    @abstractmethod\n    def preload(self, context): ...\n'
    )})
    assert state.loaded and not state.lifecycle
    names = _package_module_names(str(root), 'test/retained-parent')
    assert names
    with monkeypatch.context() as blocked:
        blocked.setattr(preload, '_remove_editor_contribution_owner', lambda *args, **kwargs: False)
        with pytest.raises(RuntimeError, match='refused to close'):
            manager.shutdown()
        assert commands.get('parent.retained') is not None
        assert _package_module_names(str(root), 'test/retained-parent') == names
    manager.shutdown()
    assert commands.get('parent.retained') is None
    assert _package_module_names(str(root), 'test/retained-parent') == ()
