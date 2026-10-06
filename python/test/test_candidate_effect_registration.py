import importlib.util
import sys
import types

import pytest

from infernux.engine.candidate_import import CandidateImportTransaction
from infernux.engine import project_context
from infernux.renderstack import render_effect_compiler as compiler


@pytest.fixture
def effect_project(tmp_path, monkeypatch):
    assets = tmp_path / 'Assets'
    assets.mkdir()
    previous = project_context.get_project_root()
    project_context.set_project_root(str(tmp_path))
    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, '_FEATURES', dict(compiler._FEATURES))
    try:
        yield assets
    finally:
        project_context.set_project_root(previous)


def declaration(type_id, name='Before', style='decorator'):
    decorator = f'@inx.renderstack.render_effect_feature({type_id!r})\n' if style == 'decorator' else ''
    registration = f'\ninx.renderstack.register_render_effect_feature({type_id!r}, ProbeEffect)\n' if style == 'function' else ''
    return f'''import infernux as inx
{decorator}class ProbeEffect(inx.renderstack.FullScreenEffect):
    name = {name!r}
    injection_point = "final"
    def setup_passes(self, graph, bus):
        pass
{registration}'''


def published_source(assets, monkeypatch, name, source):
    path = assets.joinpath(*name.split('.')).with_suffix('.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return path, module


@pytest.mark.parametrize('style', ['decorator', 'function'])
def test_rejected_candidate_effect_does_not_leak_before_or_after_rollback(effect_project, monkeypatch, style):
    type_id = 'tests.candidate.rejected.' + style
    source = declaration(type_id, style=style)
    path, original = published_source(effect_project, monkeypatch, 'candidate_rejected_' + style, source)
    accepted = compiler._FEATURES[type_id]
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    bad = source.replace("'Before'", "'After'") + '\nraise RuntimeError("reject candidate")\n'
    transaction = CandidateImportTransaction()
    transaction.register(original.__name__, str(path), source=bad)
    try:
        with pytest.raises(RuntimeError, match='reject candidate'):
            transaction.load(original.__name__)
        assert compiler._FEATURES[type_id] is accepted
        assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
        assert sys.modules[original.__name__] is original
    finally:
        transaction.rollback()
    assert compiler._FEATURES[type_id] is accepted


@pytest.mark.parametrize('style', ['decorator', 'function'])
def test_candidate_feature_is_private_until_commit_and_owner_rollback_restores_it(effect_project, monkeypatch, style):
    type_id = 'tests.candidate.replaced.' + style
    source = declaration(type_id, style=style)
    path, original = published_source(effect_project, monkeypatch, 'candidate_replaced_' + style, source)
    accepted = compiler._FEATURES[type_id]
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    transaction = CandidateImportTransaction()
    transaction.register(original.__name__, str(path), source=source.replace("'Before'", "'After'"))
    try:
        candidate = transaction.load(original.__name__)
        assert compiler._FEATURES[type_id] is accepted
        assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
        transaction.commit()
        assert compiler._FEATURES[type_id].effect_class is candidate.ProbeEffect
        assert sys.modules[original.__name__] is candidate
        assert compiler.RenderEffectArtifactRegistry.topology_generation() > generation
        # Another source registered after this publication is not part of its
        # before-image and must survive the owner's later rejection.
        published_source(effect_project, monkeypatch, 'unrelated_effect', declaration('tests.candidate.unrelated'))
        unrelated_feature = compiler._FEATURES['tests.candidate.unrelated']
        transaction.rollback()
        assert compiler._FEATURES[type_id] is accepted
        assert sys.modules[original.__name__] is original
        assert compiler._FEATURES['tests.candidate.unrelated'] is unrelated_feature
    finally:
        transaction.rollback()


def test_successful_empty_source_retires_feature_only_on_commit_and_can_rollback(effect_project, monkeypatch):
    type_id = 'tests.candidate.removed'
    path, original = published_source(effect_project, monkeypatch, 'candidate_removed', declaration(type_id))
    accepted = compiler._FEATURES[type_id]
    transaction = CandidateImportTransaction()
    transaction.register(original.__name__, str(path), source='# declaration deliberately removed\n')
    try:
        transaction.load(original.__name__)
        assert compiler._FEATURES[type_id] is accepted
        transaction.commit()
        assert type_id not in compiler._FEATURES
        transaction.rollback()
        assert compiler._FEATURES[type_id] is accepted
    finally:
        transaction.rollback()


def test_candidate_dependency_feature_lookup_uses_transaction_private_declaration(effect_project, monkeypatch):
    type_id = 'tests.candidate.dependency'
    path, original = published_source(effect_project, monkeypatch, 'effect_dependency', declaration(type_id))
    accepted = compiler._FEATURES[type_id]
    root = effect_project / 'root.py'
    root.write_text('import effect_dependency\nimport infernux as inx\nFOUND = inx.renderstack.get_render_effect_feature("tests.candidate.dependency")\n', encoding='utf-8')
    transaction = CandidateImportTransaction()
    transaction.register(original.__name__, str(path), source=declaration(type_id, 'After'))
    transaction.register('effect_root', str(root))
    try:
        candidate = transaction.load('effect_root')
        assert candidate.FOUND.effect_class is candidate.effect_dependency.ProbeEffect
        assert compiler._FEATURES[type_id] is accepted
        transaction.commit()
        assert compiler._FEATURES[type_id] is candidate.FOUND
    finally:
        transaction.rollback()


def test_feature_registration_is_restored_when_module_publication_fails(effect_project, monkeypatch):
    type_id = 'tests.candidate.failed_commit'
    path, original = published_source(effect_project, monkeypatch, 'owner.effect', declaration(type_id))
    accepted = compiler._FEATURES[type_id]

    class RejectingParent(types.ModuleType):
        def __setattr__(self, name, value):
            if name == 'effect':
                raise RuntimeError('reject parent publication')
            super().__setattr__(name, value)

    parent = RejectingParent('owner')
    parent.__path__ = [str(effect_project)]
    monkeypatch.setitem(sys.modules, 'owner', parent)
    transaction = CandidateImportTransaction()
    transaction.register(original.__name__, str(path), source=declaration(type_id, 'After'))
    try:
        transaction.load(original.__name__)
        with pytest.raises(RuntimeError, match='reject parent publication'):
            transaction.commit()
        assert compiler._FEATURES[type_id] is accepted
        assert sys.modules[original.__name__] is original
    finally:
        transaction.rollback()


def test_initial_script_failure_does_not_publish_effect_declaration(effect_project):
    from infernux.components.script_loader import _load_script_module

    type_id = 'tests.candidate.initial_failure'
    path = effect_project / 'initial_failure.py'
    source = declaration(type_id) + '\nraise RuntimeError("initial load rejected")\n'
    path.write_text(source, encoding='utf-8')
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    with pytest.raises(RuntimeError, match='initial load rejected'):
        _load_script_module(str(path), 'initial_failure', source=source)
    assert type_id not in compiler._FEATURES
    assert 'initial_failure' not in sys.modules
    assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation


@pytest.mark.parametrize('fail_commit', [False, True])
def test_component_owner_commits_or_rejects_effect_with_the_class_body(effect_project, monkeypatch, fail_commit):
    from infernux.components import script_loader

    type_id = 'tests.candidate.component_owner'
    source = declaration(type_id) + '''
class EffectOwnerProbe(inx.InxComponent):
    def marker(self):
        return 'Before'
'''
    path, original = published_source(effect_project, monkeypatch, 'effect_owner', source)
    accepted = compiler._FEATURES[type_id]
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    transaction = script_loader.stage_component_body_reload_batch((
        script_loader.ComponentBodyReloadRequest(
            str(path), target_types=(original.EffectOwnerProbe,),
            source=source.replace("'Before'", "'After'"),
        ),
    ))
    assert compiler._FEATURES[type_id] is accepted
    if fail_commit:
        real_apply = script_loader._apply_component_body_patch_plans

        def reject_after_body(plans):
            real_apply(plans)
            raise RuntimeError('reject owner body publication')

        monkeypatch.setattr(script_loader, '_apply_component_body_patch_plans', reject_after_body)
    try:
        if fail_commit:
            with pytest.raises(RuntimeError, match='reject owner body publication'):
                transaction.commit()
            assert compiler._FEATURES[type_id] is accepted
            assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
        else:
            transaction.commit()
            assert compiler._FEATURES[type_id].effect_class.name == 'After'
            assert original.EffectOwnerProbe().marker() == 'After'
        transaction.rollback()
        assert compiler._FEATURES[type_id] is accepted
        assert original.EffectOwnerProbe().marker() == 'Before'
        assert sys.modules[original.__name__] is original
    finally:
        transaction.rollback()


def test_candidate_missing_feature_does_not_run_global_discovery(effect_project, monkeypatch):
    from infernux.renderstack import discovery

    path = effect_project / 'missing_feature.py'
    path.write_text('import infernux as inx\ninx.renderstack.get_render_effect_feature("tests.missing")\n', encoding='utf-8')

    def unexpected_discovery():
        raise AssertionError('candidate lookup escaped to global source discovery')

    monkeypatch.setattr(discovery, 'discover_effect_features', unexpected_discovery)
    transaction = CandidateImportTransaction()
    transaction.register('missing_feature', str(path))
    try:
        with pytest.raises(compiler.RenderEffectCompileError, match='unknown render effect feature'):
            transaction.load('missing_feature')
    finally:
        transaction.rollback()


def test_component_prepare_rejection_discards_staged_effect(effect_project, monkeypatch):
    from infernux.components import script_loader

    type_id = 'tests.candidate.prepare_rejected'
    source = declaration(type_id) + '\nclass PrepareEffectOwner(inx.InxComponent):\n    pass\n'
    path, original = published_source(effect_project, monkeypatch, 'prepare_effect_owner', source)
    accepted = compiler._FEATURES[type_id]
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    with pytest.raises(script_loader.ScriptReloadRejected, match='no component classes'):
        script_loader.stage_component_body_reload_batch((
            script_loader.ComponentBodyReloadRequest(
                str(path), target_types=(original.PrepareEffectOwner,),
                source=declaration(type_id, 'After'),
            ),
        ))
    assert compiler._FEATURES[type_id] is accepted
    assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
    assert sys.modules[original.__name__] is original
