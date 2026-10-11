"""Static preload discovery follows Python's local import bindings."""
from pathlib import Path

import pytest

from infernux.plugins import PluginManager
from infernux.plugins.preload import _read_declarations


@pytest.mark.parametrize('statement,base,expected', [
    ('import infernux.lifecycle', 'infernux.lifecycle.InxPreload', 'infernux.lifecycle.InxPreload'),
    ('import infernux.lifecycle as lifecycle', 'lifecycle.InxPreload', 'infernux.lifecycle.InxPreload'),
    ('import infernux', 'infernux.InxPreload', 'infernux.InxPreload'),
    ('import infernux as inx', 'inx.InxPreload', 'infernux.InxPreload'),
    ('from infernux.lifecycle import InxPreload as Base', 'Base', 'infernux.lifecycle.InxPreload'),
    ('import game.core.foundation', 'game.core.foundation.Base', 'game.core.foundation.Base'),
    ('import game.core.foundation as foundation', 'foundation.Base', 'game.core.foundation.Base'),
    ('import game.core.foundation, infernux.lifecycle', 'infernux.lifecycle.InxPreload', 'infernux.lifecycle.InxPreload'),
])
def test_preload_base_uses_the_python_local_binding(tmp_path, statement, base, expected):
    path = tmp_path / 'Assets' / 'startup.py'
    path.parent.mkdir()
    path.write_text(statement + '\nclass Startup(' + base + '): pass\n', encoding='utf-8')
    declaration, = _read_declarations(str(path), str(tmp_path))
    assert declaration.bases == (expected,)


@pytest.mark.parametrize('style', ['dotted', 'alias', 'from'])
@pytest.mark.parametrize('inherited', [False, True])
def test_real_preload_discovery_and_reload_run_once_per_candidate(tmp_path, monkeypatch, style, inherited):
    root = tmp_path / 'Project'
    assets = root / 'Assets'
    assets.mkdir(parents=True)
    (root / 'ProjectSettings').mkdir()
    monkeypatch.setenv('INFERNUX_PACKAGE_CACHE_ROOT', str(tmp_path / 'cache'))
    imports = {
        'dotted': ('import infernux.lifecycle', 'infernux.lifecycle.InxPreload'),
        'alias': ('import infernux.lifecycle as life', 'life.InxPreload'),
        'from': ('from infernux.lifecycle import InxPreload', 'InxPreload'),
    }
    statement, base = imports[style]
    if inherited:
        package = assets / 'preload_import_contract'
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'foundation.py').write_text(
            statement + '\nfrom abc import abstractmethod\n'
            + 'class Foundation(' + base + '):\n'
            + '    @abstractmethod\n    def configure(self): ...\n'
            + '    def preload(self, context): ...\n', encoding='utf-8')
        statement, base = {
            'dotted': ('import preload_import_contract.foundation', 'preload_import_contract.foundation.Foundation'),
            'alias': ('import preload_import_contract.foundation as foundation', 'foundation.Foundation'),
            'from': ('from preload_import_contract.foundation import Foundation', 'Foundation'),
        }[style]
    (assets / 'startup.py').write_text(
        statement + '\nfrom pathlib import Path\nclass Startup(' + base + '):\n'
        + '    def configure(self): pass\n'
        + '    def preload(self, context):\n'
        + '        with Path(context.project_root, "started.txt").open("a") as output:\n'
        + '            output.write("loaded\\n")\n', encoding='utf-8')
    marker = root / 'ordinary-imported.txt'
    (assets / 'ordinary.py').write_text(
        'from pathlib import Path\n' + f'Path({str(marker)!r}).write_text("wrong")\n', encoding='utf-8')
    manager = PluginManager(str(root))
    try:
        for count in (1, 2):
            manager.reload_all()
            states = manager.preloads.snapshots()
            assert len(states) == 1 and states[0]['loaded']
            assert states[0]['type_name'] == 'Startup'
            assert not manager.preloads.failures
            assert (root / 'started.txt').read_text(encoding='utf-8').splitlines() == ['loaded'] * count
            assert not marker.exists()
    finally:
        manager.shutdown()
