import os
import sys
import time
import types

import pytest

import infernux.renderstack.discovery as discovery
from infernux.engine.path_utils import path_key
from infernux.engine.project_context import get_project_root, set_project_root


@pytest.fixture
def catalog_project(tmp_path):
    assets = tmp_path / 'Assets'
    assets.mkdir()
    previous = get_project_root()
    set_project_root(str(tmp_path))
    discovery.invalidate_discovery_cache()
    try:
        yield assets
    finally:
        set_project_root(previous)
        discovery.invalidate_discovery_cache()


def write_source(path, source, revision):
    path.write_text(source, encoding='utf-8')
    stamp = time.time() + revision
    os.utime(path, (stamp, stamp))
    discovery.invalidate_discovery_cache()


@pytest.mark.parametrize('base_name,discover', [('RenderPipeline', discovery.discover_pipelines), ('RenderPass', discovery.discover_passes)])
def test_catalog_revisits_known_class_and_retires_previous_name(catalog_project, base_name, discover):
    path = catalog_project / 'renamed.py'
    source = f'from infernux.renderstack import {base_name}\nclass CatalogPublished{base_name}({base_name}):\n    name = "Publication Old {base_name}"\n'
    write_source(path, source, 0)
    old_name, new_name = f'Publication Old {base_name}', f'Publication New {base_name}'
    retained = discover()[old_name]
    write_source(path, source.replace(old_name, new_name), 3)
    current = discover()
    assert new_name in current
    assert old_name not in current
    assert current[new_name] is not retained


@pytest.mark.parametrize('base_name,discover', [('RenderPipeline', discovery.discover_pipelines), ('RenderPass', discovery.discover_passes)])
def test_catalog_excludes_removed_declaration_in_existing_file(catalog_project, base_name, discover):
    path = catalog_project / 'removed_declaration.py'
    name = f'Removed Declaration {base_name}'
    source = f'from infernux.renderstack import {base_name}\nclass CatalogRemoved{base_name}({base_name}):\n    name = "{name}"\n'
    write_source(path, source, 0)
    retained = discover()[name]
    write_source(path, '# source exists but no longer defines the class\n', 3)
    assert name not in discover()
    assert retained.name == name


@pytest.mark.parametrize('base_name,discover', [('RenderPipeline', discovery.discover_pipelines), ('RenderPass', discovery.discover_passes)])
def test_catalog_never_admits_class_from_rejected_module(catalog_project, base_name, discover):
    path = catalog_project / 'rejected_candidate.py'
    accepted_name, rejected_name = f'Accepted Publication {base_name}', f'Rejected Publication {base_name}'
    source = f'from infernux.renderstack import {base_name}\nclass CatalogCandidate{base_name}({base_name}):\n    name = "{accepted_name}"\n'
    write_source(path, source, 0)
    accepted = discover()[accepted_name]
    write_source(path, source.replace(accepted_name, rejected_name) + 'raise RuntimeError("candidate publication rejected")\n', 3)
    current = discover()
    # Retain the partial candidate explicitly; GC must not decide validity.
    rejected = next(cls for cls in accepted.__bases__[0].__subclasses__() if cls.name == rejected_name)
    assert rejected_name not in current
    assert current[accepted_name] is accepted
    assert 'candidate publication rejected' in discovery.discovery_import_failures()[path_key(str(path))]
    assert rejected.name == rejected_name


@pytest.mark.parametrize('base_name,discover', [('RenderPipeline', discovery.discover_pipelines), ('RenderPass', discovery.discover_passes)])
def test_deleted_rejected_source_retires_accepted_class_and_diagnostic(catalog_project, base_name, discover):
    path = catalog_project / 'deleted_rejected.py'
    name = f'Deleted Rejected {base_name}'
    source = f'from infernux.renderstack import {base_name}\nclass DeletedRejected{base_name}({base_name}):\n    name = "{name}"\n'
    write_source(path, source, 0)
    retained = discover()[name]
    write_source(path, source + 'raise RuntimeError("rejected before deletion")\n', 3)
    assert discover()[name] is retained
    assert path_key(str(path)) in discovery.discovery_import_failures()
    path.unlink()
    discovery.invalidate_discovery_cache()
    assert name not in discover()
    assert path_key(str(path)) not in discovery.discovery_import_failures()


@pytest.mark.parametrize('nested', [False, True])
def test_live_class_belongs_to_current_published_namespace(tmp_path, monkeypatch, nested):
    source = tmp_path / 'published.py'
    source.write_text('# present\n', encoding='utf-8')
    module_name = '_catalog_published_namespace_test'
    old = types.ModuleType(module_name)
    old.__file__ = str(source)
    qualifier = 'Namespace.Pipeline' if nested else 'Pipeline'
    retired = type('Pipeline', (), {'__module__': module_name, '__qualname__': qualifier})
    current_class = type('Pipeline', (), {'__module__': module_name, '__qualname__': qualifier})
    current = types.ModuleType(module_name)
    current.__file__ = str(source)
    if nested:
        current.Namespace = types.SimpleNamespace(Pipeline=current_class)
    else:
        current.Pipeline = current_class
    monkeypatch.setitem(sys.modules, module_name, current)
    assert discovery._is_live_class(current_class)
    assert not discovery._is_live_class(retired)


def test_public_factory_class_alias_remains_discoverable(tmp_path, monkeypatch):
    source = tmp_path / 'factory.py'
    source.write_text('# present\n', encoding='utf-8')
    module_name = '_catalog_public_factory_test'
    module = types.ModuleType(module_name)
    module.__file__ = str(source)
    generated = type('GeneratedPipeline', (), {'__module__': module_name, '__qualname__': 'factory.<locals>.GeneratedPipeline'})
    module.ExportedPipeline = generated
    monkeypatch.setitem(sys.modules, module_name, module)
    assert discovery._is_live_class(generated)


@pytest.mark.parametrize('base_name,discover,kind', [
    ('RenderPipeline', discovery.discover_pipelines, 'pipeline'),
    ('RenderPass', discovery.discover_passes, 'pass'),
])
def test_duplicate_catalog_names_have_no_implicit_winner(catalog_project, base_name, discover, kind):
    name = f'Duplicate Catalog {base_name}'
    for filename, classname in [('first.py', 'FirstProvider'), ('second.py', 'SecondProvider')]:
        write_source(catalog_project / filename,
                     f'from infernux.renderstack import {base_name}\nclass {classname}({base_name}):\n    name = {name!r}\n', 0)
    assert name not in discover()
    conflicts = discovery.discovery_name_conflicts(kind)
    candidates = conflicts[name]
    assert len(candidates) == 2 and candidates == tuple(sorted(candidates))
    assert any('first.py' in item and 'FirstProvider' in item for item in candidates)
    assert any('second.py' in item and 'SecondProvider' in item for item in candidates)
    conflicts.clear()
    assert discovery.discovery_name_conflicts(kind)[name] == candidates
    assert not discovery.discovery_import_failures()


@pytest.mark.parametrize('repair', ['rename', 'delete'])
def test_duplicate_pipeline_name_recovers_from_authoritative_source(catalog_project, repair):
    name = 'Repair Duplicate Pipeline ' + repair
    first = catalog_project / 'first.py'
    second = catalog_project / 'second.py'
    template = 'from infernux.renderstack import RenderPipeline\nclass {cls}(RenderPipeline):\n    name = {name!r}\n'
    write_source(first, template.format(cls='RepairFirst', name=name), 0)
    accepted = discovery.discover_pipelines()[name]
    write_source(second, template.format(cls='RepairSecond', name=name), 0)
    assert name not in discovery.discover_pipelines()
    if repair == 'rename':
        write_source(second, template.format(cls='RepairSecond', name=name + ' Renamed'), 3)
    else:
        second.unlink()
        discovery.invalidate_discovery_cache()
    assert discovery.discover_pipelines()[name] is accepted
    assert name not in discovery.discovery_name_conflicts()
    if repair == 'rename':
        assert name + ' Renamed' in discovery.discover_pipelines()


def test_duplicate_default_pipeline_selection_is_rejected(catalog_project):
    from infernux.renderstack import RenderStack

    write_source(catalog_project / 'default_collision.py',
                 'from infernux.renderstack import RenderPipeline\nclass ProjectDefault(RenderPipeline):\n    name = "Default Forward"\n', 0)
    stack = RenderStack()
    try:
        with pytest.raises(RuntimeError, match='ambiguous.*Default Forward'):
            stack._create_pipeline()
        assert stack.pipeline_class_name == 'Default Forward'
    finally:
        stack.on_destroy()


def test_alias_import_does_not_duplicate_one_catalog_declaration(catalog_project, monkeypatch):
    import importlib.util

    name = 'One Physical Pipeline Declaration'
    path = catalog_project / 'physical.py'
    write_source(path, 'from infernux.renderstack import RenderPipeline\nclass PhysicalPipeline(RenderPipeline):\n    name = '+repr(name)+'\n', 0)
    canonical = discovery.discover_pipelines()[name]
    spec = importlib.util.spec_from_file_location('_audit_alias_import', path)
    alias = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, alias)
    spec.loader.exec_module(alias)
    assert alias.PhysicalPipeline is not canonical
    discovery.invalidate_discovery_cache()
    assert discovery.discover_pipelines()[name] is canonical
    assert name not in discovery.discovery_name_conflicts()


def test_catalog_does_not_include_the_previous_projects_retained_class(catalog_project):
    name = 'Previous Project Retained Pipeline'
    path = catalog_project / 'previous.py'
    write_source(path, 'from infernux.renderstack import RenderPipeline\nclass PreviousProject(RenderPipeline):\n    name = '+repr(name)+'\n', 0)
    retained = discovery.discover_pipelines()[name]
    next_project = catalog_project.parent / 'NextProject'
    (next_project / 'Assets').mkdir(parents=True)
    set_project_root(str(next_project))
    discovery.invalidate_discovery_cache()
    assert name not in discovery.discover_pipelines()
    assert retained.name == name


def test_compiled_player_catalog_rejects_duplicate_names(catalog_project, monkeypatch):
    import py_compile
    from infernux.renderstack import RenderStack

    monkeypatch.setenv('_INFERNUX_PLAYER_MODE', '1')
    name = 'Compiled Player Duplicate'
    for index in range(2):
        source = catalog_project.parent / f'author_{index}.py'
        source.write_text('from infernux.renderstack import RenderPipeline\nclass CompiledProvider(RenderPipeline):\n    name = '+repr(name)+'\n', encoding='utf-8')
        py_compile.compile(str(source), cfile=str(catalog_project / f'provider_{index}.pyc'), doraise=True)
    assert name not in discovery.discover_pipelines()
    assert len(discovery.discovery_name_conflicts()[name]) == 2
    stack = RenderStack()
    stack.pipeline_class_name = name
    try:
        with pytest.raises(RuntimeError, match='ambiguous.*Compiled Player Duplicate'):
            stack._create_pipeline()
        assert stack.pipeline_class_name == name
    finally:
        stack.on_destroy()


def test_inspector_exposes_unselected_pipeline_name_conflicts(catalog_project):
    from infernux.engine.ui.inspector_renderstack import _pipeline_name_conflict_messages

    name = 'Unselected Conflict'
    for index in range(2):
        write_source(catalog_project / f'conflict_{index}.py',
                     'from infernux.renderstack import RenderPipeline\nclass ConflictingProvider(RenderPipeline):\n    name = '+repr(name)+'\n', 0)
    message, = _pipeline_name_conflict_messages()
    assert name in message and 'conflict_0.py' in message and 'conflict_1.py' in message
