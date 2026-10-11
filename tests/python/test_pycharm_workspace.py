"""Opening scripts may create missing IDE helpers, never reset authored files."""
import xml.etree.ElementTree as ET

import pytest

from infernux.engine.python_abi import PYTHON_RUNTIME_DIRECTORY
from infernux.engine.ui import project_utils


FILES = ('.idea/modules.xml', '.idea/misc.xml', '.idea/project.iml',
         '.idea/.gitignore', 'PYCHARM_SETUP.zh-CN.en.md')


def _project(tmp_path):
    script = tmp_path / 'Assets/Puzzle.py'
    script.parent.mkdir()
    script.write_text('door_open = False\n', encoding='utf-8')
    runtime = tmp_path / '.runtime' / PYTHON_RUNTIME_DIRECTORY / 'python.exe'
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b'never executed')
    return script


@pytest.mark.parametrize('entrypoint', ['direct', 'project'])
@pytest.mark.parametrize('missing', FILES + (None,))
def test_open_pycharm_only_creates_missing_files(tmp_path, monkeypatch, entrypoint, missing):
    script = _project(tmp_path)
    assert project_utils._ensure_pycharm_project_files(str(tmp_path))
    original = {}
    for filename in FILES:
        path = tmp_path / filename
        if filename == missing:
            path.unlink()
        else:
            # Author comments, encoding, whitespace and file timestamps are owned by the team.
            content = ('\ufeff<!-- keep authored 中文 -->\r\n' + path.read_text(encoding='utf-8')).encode('utf-8')
            path.write_bytes(content)
            original[path] = (content, path.stat().st_mtime_ns)
    launches = []
    monkeypatch.setattr(project_utils, '_find_pycharm_executable', lambda: 'pycharm.exe')
    monkeypatch.setattr(project_utils, 'get_ide', lambda: 'pycharm')
    monkeypatch.setattr(project_utils, 'detect_available_ides', lambda **_: ['pycharm'])
    monkeypatch.setattr('subprocess.Popen', lambda args, **kwargs: launches.append(args))
    opener = project_utils.open_in_pycharm if entrypoint == 'direct' else project_utils.open_file_with_system
    assert opener(str(script), line=17, project_root=str(tmp_path))
    assert len(launches) == 1
    assert launches[0][-2:] == ['--line', '17']
    assert str(script) in launches[0] and str(tmp_path) in launches[0]
    assert all(path.read_bytes() == payload and path.stat().st_mtime_ns == stamp
               for path, (payload, stamp) in original.items())
    assert all((tmp_path / filename).is_file() for filename in FILES)
    before = [(tmp_path / name).read_bytes() for name in FILES]
    assert opener(str(script), project_root=str(tmp_path))
    assert [(tmp_path / name).read_bytes() for name in FILES] == before
    assert str(tmp_path / FILES[-1]) not in launches[-1]


def test_generated_pycharm_paths_resolve_from_the_actual_module_directory(tmp_path):
    _project(tmp_path)
    assert project_utils._ensure_pycharm_project_files(str(tmp_path))
    module = tmp_path / '.idea/project.iml'
    root = ET.parse(module).getroot()
    content = root.find('./component/content')

    def local_path(url):
        assert url.startswith('file://$MODULE_DIR$')
        return (module.parent / url.removeprefix('file://$MODULE_DIR$').lstrip('/')).resolve()

    assert local_path(content.attrib['url']) == tmp_path.resolve()
    assert local_path(content.find('sourceFolder').attrib['url']) == (tmp_path / 'Assets').resolve()
    excludes = {local_path(item.attrib['url']) for item in content.findall('excludeFolder')}
    assert (tmp_path / '.runtime').resolve() in excludes
    guide = (tmp_path / FILES[-1]).read_text(encoding='utf-8')
    assert str(tmp_path) not in guide
    assert f'.runtime/{PYTHON_RUNTIME_DIRECTORY}/python.exe' in guide


def test_pycharm_preserves_a_file_created_by_the_ide_during_initialization(tmp_path, monkeypatch):
    import builtins

    _project(tmp_path)
    target = tmp_path / '.idea/misc.xml'
    authored = b'<project><component name="ProjectRootManager" project-jdk-name="TeamSDK" /></project>'
    original_open = builtins.open

    def concurrent_ide_save(path, mode='r', *args, **kwargs):
        if str(path) == str(target) and mode == 'x':
            target.write_bytes(authored)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', concurrent_ide_save)
    assert project_utils._ensure_pycharm_project_files(str(tmp_path))
    assert target.read_bytes() == authored


def test_pycharm_config_write_failure_prevents_launch(tmp_path, monkeypatch):
    import builtins

    script = _project(tmp_path)
    original_open = builtins.open
    launches = []

    def reject_template_write(path, mode='r', *args, **kwargs):
        if str(path).endswith('modules.xml') and mode == 'x':
            raise PermissionError('IDE project is read-only')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', reject_template_write)
    monkeypatch.setattr(project_utils, '_find_pycharm_executable', lambda: 'pycharm.exe')
    monkeypatch.setattr('subprocess.Popen', lambda args, **kwargs: launches.append(args))
    with pytest.raises(PermissionError, match='read-only'):
        project_utils.open_in_pycharm(str(script), project_root=str(tmp_path))
    assert not launches
    assert not list((tmp_path / '.idea').iterdir())
