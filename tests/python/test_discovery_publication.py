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


@pytest.mark.parametrize('compiled', [False, True])
def test_same_filename_sources_cannot_overwrite_each_others_namespace(catalog_project, monkeypatch, compiled):
    import py_compile

    # Simulate an address-suffix collision: the old suffix has only 16 bits.
    monkeypatch.setattr(discovery, 'id', lambda _value: 0, raising=False)
    paths = []
    for index in range(2):
        directory = catalog_project / f'branch_{index}'
        directory.mkdir()
        source = directory / 'shared.py'
        source.write_text('from infernux.renderstack import RenderPipeline\nclass Shared(RenderPipeline):\n    name = '+repr(f'Independent Branch {index}')+'\n', encoding='utf-8')
        if compiled:
            target = directory / 'shared.pyc'
            py_compile.compile(str(source), cfile=str(target), doraise=True)
            source.unlink()
        else:
            target = source
        paths.append(target)
    catalog = discovery.discover_pipelines()
    assert {'Independent Branch 0', 'Independent Branch 1'} <= catalog.keys()
    retained = catalog['Independent Branch 1']
    assert catalog['Independent Branch 0'].__module__ != retained.__module__
    paths[0].unlink()
    discovery.invalidate_discovery_cache()
    assert 'Independent Branch 0' not in discovery.discover_pipelines()
    assert discovery.discover_pipelines()['Independent Branch 1'] is retained


@pytest.mark.parametrize('base_name,discover', [('RenderPipeline', discovery.discover_pipelines), ('RenderPass', discovery.discover_passes)])
def test_same_second_same_size_save_reads_current_source_not_cached_bytecode(catalog_project, base_name, discover):
    path = catalog_project / 'rapid.py'
    old_name, new_name = f'Rapid Old {base_name}', f'Rapid New {base_name}'
    source = f'from infernux.renderstack import {base_name}\nclass Rapid({base_name}):\n    name = {old_name!r}\n'
    stamp = int(time.time()) * 1_000_000_000
    path.write_text(source, encoding='utf-8')
    os.utime(path, ns=(stamp + 100_000_000, stamp + 100_000_000))
    retained = discover()[old_name]
    path.write_text(source.replace(old_name, new_name), encoding='utf-8')
    os.utime(path, ns=(stamp + 200_000_000, stamp + 200_000_000))
    discovery.invalidate_discovery_cache()
    current = discover()
    assert new_name in current and old_name not in current
    assert current[new_name] is not retained


def test_file_size_change_invalidates_import_even_with_preserved_mtime(catalog_project):
    path = catalog_project / 'preserved_time.py'
    source = 'from infernux.renderstack import RenderPipeline\nclass PreservedTime(RenderPipeline):\n    name = "Size Before"\n'
    path.write_text(source, encoding='utf-8')
    stat = path.stat()
    assert 'Size Before' in discovery.discover_pipelines()
    path.write_text(source.replace('Size Before', 'Size After With A Longer Name'), encoding='utf-8')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    discovery.invalidate_discovery_cache()
    current = discovery.discover_pipelines()
    assert 'Size After With A Longer Name' in current and 'Size Before' not in current


def test_discovered_source_supports_python_dataclass_declarations(catalog_project):
    path = catalog_project / 'dataclass_pipeline.py'
    path.write_text('''from __future__ import annotations
from dataclasses import dataclass
from infernux.renderstack import RenderPipeline
@dataclass
class Settings:
    gain: float = 1.0
class DataclassPipeline(RenderPipeline):
    name = "Dataclass Pipeline"
    settings = Settings()
''', encoding='utf-8')
    catalog = discovery.discover_pipelines()
    assert 'Dataclass Pipeline' in catalog, discovery.discovery_import_failures()
    assert catalog['Dataclass Pipeline']().settings.gain == 1.0


@pytest.mark.parametrize('declaration', ['decorator', 'function'])
def test_failed_effect_source_cannot_publish_replacement_registration(catalog_project, monkeypatch, declaration):
    from infernux.renderstack import render_effect_compiler as compiler

    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, '_FEATURES', dict(compiler._FEATURES))
    path = catalog_project / 'transaction_feature.py'
    decorator = '@inx.renderstack.render_effect_feature("tests.discovery.transaction")\n' if declaration == 'decorator' else ''
    registration = '\ninx.renderstack.register_render_effect_feature("tests.discovery.transaction", TransactionFeature)\n' if declaration == 'function' else ''
    source = f'''import infernux as inx
{decorator}class TransactionFeature(inx.renderstack.FullScreenEffect):
    name = "Transaction Before"
    injection_point = "final"
    def setup_passes(self, graph, bus):
        pass
{registration}'''
    write_source(path, source, 0)
    discovery.discover_effect_features()
    accepted = compiler.get_render_effect_feature('tests.discovery.transaction')
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    write_source(path, source.replace('Transaction Before', 'Transaction After') + '\nraise RuntimeError("reject candidate")\n', 3)
    discovery.discover_effect_features()
    assert compiler.get_render_effect_feature('tests.discovery.transaction') is accepted
    assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
    assert sys.modules[accepted.effect_class.__module__].TransactionFeature is accepted.effect_class
    assert 'reject candidate' in discovery.discovery_import_failures()[path_key(path)]
    write_source(path, source.replace('Transaction Before', 'Transaction After'), 6)
    discovery.discover_effect_features()
    repaired = compiler.get_render_effect_feature('tests.discovery.transaction')
    assert repaired is not accepted and repaired.effect_class.name == 'Transaction After'
    assert compiler.RenderEffectArtifactRegistry.topology_generation() > generation


def test_known_catalog_source_stays_classified_after_rejected_import(catalog_project):
    path = catalog_project / 'known_failure.py'
    source = 'from infernux.renderstack import RenderPipeline\nclass KnownFailure(RenderPipeline):\n    name = "Known Failure"\n'
    write_source(path, source, 0)
    retained = discovery.discover_pipelines()['Known Failure']
    write_source(path, source + 'raise RuntimeError("reject source")\n', 3)
    assert discovery.discover_pipelines()['Known Failure'] is retained
    write_source(path, '# removing the final declaration after a failed edit\n', 6)
    assert discovery.script_may_affect_pipeline_catalog(str(path))
    assert 'Known Failure' not in discovery.discover_pipelines()


def test_reentrant_catalog_read_cannot_publish_inflight_or_erase_restored_class(catalog_project, monkeypatch):
    from infernux.renderstack import render_effect_compiler as compiler

    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, '_FEATURES', dict(compiler._FEATURES))
    path = catalog_project / 'reentrant.py'
    source = '''import infernux as inx
from infernux.renderstack.discovery import discover_pipelines
class ReentrantPipeline(inx.renderstack.RenderPipeline):
    name = "Reentrant Pipeline"
@inx.renderstack.render_effect_feature("tests.discovery.reentrant")
class ReentrantEffect(inx.renderstack.FullScreenEffect):
    name = "Reentrant Effect"
    injection_point = "final"
    def setup_passes(self, graph, bus):
        pass
'''
    write_source(path, source, 0)
    retained = discovery.discover_pipelines()['Reentrant Pipeline']
    write_source(path, source + 'visible_during_execution = discover_pipelines()\nraise RuntimeError("reject reentrant")\n', 3)
    discovery.discover_effect_features()
    assert discovery.discover_pipelines()['Reentrant Pipeline'] is retained
    write_source(path, source + 'visible_during_execution = discover_pipelines()\n', 6)
    discovery.discover_effect_features()
    module = sys.modules[discovery.discover_pipelines()['Reentrant Pipeline'].__module__]
    assert 'Reentrant Pipeline' not in module.visible_during_execution


def test_successful_dependency_registration_survives_importing_source_failure(catalog_project, monkeypatch):
    from infernux.renderstack import render_effect_compiler as compiler

    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, '_FEATURES', dict(compiler._FEATURES))
    dependency = catalog_project / '_feature_dependency.py'
    dependency.write_text('''import infernux as inx
@inx.renderstack.render_effect_feature("tests.discovery.dependency")
class Dependency(inx.renderstack.FullScreenEffect):
    name = "Dependency Effect"
    injection_point = "final"
    def setup_passes(self, graph, bus):
        pass
''', encoding='utf-8')
    path = catalog_project / 'dependent.py'
    source = 'from _feature_dependency import Dependency\nfrom infernux.renderstack import RenderPipeline\nclass Dependent(RenderPipeline):\n    name = "Dependent Pipeline"\n'
    write_source(path, source + 'raise RuntimeError("reject importer")\n', 0)
    try:
        assert 'Dependent Pipeline' not in discovery.discover_pipelines()
        accepted = compiler.get_render_effect_feature('tests.discovery.dependency')
        write_source(path, source, 3)
        assert 'Dependent Pipeline' in discovery.discover_pipelines()
        assert compiler.get_render_effect_feature('tests.discovery.dependency') is accepted
    finally:
        sys.modules.pop('_feature_dependency', None)


def test_declared_source_encoding_preserves_indirect_inheritance(catalog_project):
    base = catalog_project / 'latin_base.py'
    base.write_bytes(('''# coding: latin-1
from infernux.renderstack import RenderPipeline
class Piñata(RenderPipeline):
    name = "Encoded Base Pipeline"
''').encode('latin-1'))
    child = catalog_project / 'encoded_child.py'
    child.write_text('''from latin_base import Piñata
class EncodedChild(Piñata):
    name = "Encoded Child Pipeline"
''', encoding='utf-8')
    try:
        assert {'Encoded Base Pipeline', 'Encoded Child Pipeline'} <= discovery.discover_pipelines().keys()
        assert not discovery.discovery_import_failures()
    finally:
        sys.modules.pop('latin_base', None)


@pytest.mark.parametrize('removal', ['declaration', 'file'])
def test_accepted_effect_source_removal_retires_its_registration(catalog_project, monkeypatch, removal):
    from infernux.renderstack import render_effect_compiler as compiler

    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, '_FEATURES', dict(compiler._FEATURES))
    path = catalog_project / 'removed_feature.py'
    source = '''import infernux as inx
@inx.renderstack.render_effect_feature("tests.discovery.removed")
class RemovedEffect(inx.renderstack.FullScreenEffect):
    name = "Removed Effect"
    injection_point = "final"
    def setup_passes(self, graph, bus):
        pass
'''
    write_source(path, source, 0)
    accepted = compiler.get_render_effect_feature('tests.discovery.removed')
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    if removal == 'declaration':
        write_source(path, '# source no longer declares an effect\n', 3)
    else:
        path.unlink()
        discovery.invalidate_discovery_cache()
    with pytest.raises(compiler.RenderEffectCompileError, match='unknown render effect feature'):
        compiler.get_render_effect_feature('tests.discovery.removed')
    assert compiler.RenderEffectArtifactRegistry.topology_generation() > generation
    retired_generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    write_source(path, source, 6)
    assert compiler.get_render_effect_feature('tests.discovery.removed') is not accepted
    assert compiler.RenderEffectArtifactRegistry.topology_generation() > retired_generation


def test_discovery_reconciles_successful_registration_from_normal_python_import(catalog_project, monkeypatch):
    import importlib.util
    from infernux.renderstack import render_effect_compiler as compiler

    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, '_FEATURES', dict(compiler._FEATURES))
    path = catalog_project / 'external_feature.py'
    source = '''import infernux as inx
@inx.renderstack.render_effect_feature("tests.discovery.external")
class ExternalEffect(inx.renderstack.FullScreenEffect):
    name = "External Effect"
    injection_point = "final"
    def setup_passes(self, graph, bus):
        pass
'''
    write_source(path, source, 0)
    spec = importlib.util.spec_from_file_location('_external_registered_source', path)
    imported = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, imported)
    spec.loader.exec_module(imported)
    path.write_text('# removed after a regular Python import\n', encoding='utf-8')
    discovery.invalidate_discovery_cache()
    assert discovery.script_may_affect_pipeline_catalog(str(path))
    with pytest.raises(compiler.RenderEffectCompileError, match='unknown render effect feature'):
        compiler.get_render_effect_feature('tests.discovery.external')


def test_source_publication_reloads_content_even_when_file_stamp_is_unchanged(catalog_project):
    from infernux.renderstack import RenderStack
    from infernux.engine.resources_manager import ResourcesManager

    path = catalog_project / 'event_authority.py'
    source = 'from infernux.renderstack import RenderPipeline\nclass EventAuthority(RenderPipeline):\n    name = "Event Old"\n'
    path.write_text(source, encoding='utf-8')
    stat = path.stat()
    stack = RenderStack()
    try:
        stack._sync_pipeline_catalog()
        assert 'Event Old' in discovery.discover_pipelines()
        path.write_text(source.replace('Event Old', 'Event New'), encoding='utf-8')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        manager = ResourcesManager.__new__(ResourcesManager)
        manager._script_catalog_callbacks = [stack._on_script_catalog_changed]
        manager.notify_script_catalog_changed(str(path), 'modified')
        assert 'Event New' in discovery.discover_pipelines()
        assert 'Event Old' not in discovery.discover_pipelines()
    finally:
        stack.on_destroy()


def test_one_source_event_publishes_one_namespace_for_multiple_consumers(catalog_project):
    from infernux.engine.resources_manager import ResourcesManager

    path = catalog_project / 'multiple_consumers.py'
    source = 'from infernux.renderstack import RenderPipeline\nclass MultipleConsumers(RenderPipeline):\n    name = "Consumers Old"\n'
    write_source(path, source, 0)
    old = discovery.discover_pipelines()['Consumers Old']
    stat = path.stat()
    path.write_text(source.replace('Consumers Old', 'Consumers New'), encoding='utf-8')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    observed = []
    manager = ResourcesManager.__new__(ResourcesManager)
    manager._script_catalog_callbacks = [lambda *_: observed.append(discovery.discover_pipelines()['Consumers New']) for _ in range(2)]
    manager.notify_script_catalog_changed(str(path), 'modified')
    assert len(observed) == 2 and observed[0] is observed[1] and observed[0] is not old
