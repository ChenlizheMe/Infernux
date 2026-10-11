"""Live settings consume the same mutable resources used by gameplay scripts."""
from types import SimpleNamespace

import pytest

from test_web_renderstack_template import _web_settings_functions
from infernux.renderstack.effect_slot import EffectSlot
from infernux.renderstack.forward_parameters import DefaultForwardParameters, MSAASamples
from infernux.renderstack.render_effect import EditableRenderEffectGroup, RenderEffect
from infernux.renderstack.render_effect_asset import (
    EffectAssetReference, RenderEffectAsset, RenderEffectGroupAsset, RenderEffectGroupEntry,
)


def effect(kind='tonemapping', **parameters):
    return RenderEffect(RenderEffectAsset('infernux.post.' + kind, parameters))


def stack(*effects):
    return SimpleNamespace(pipeline=DefaultForwardParameters(), effect_slots=[
        EffectSlot(stage_id='final', effect=value) for value in effects])


def state():
    return _web_settings_functions()['_WebRenderSettingsState']()


def test_live_changes_use_revisions_and_unchanged_frames_do_not_copy_assets(monkeypatch):
    tone = effect(exposure=1.0)
    owner = stack(tone)
    monitor = state()
    assert monitor.update(owner)['tonemapping_exposure'] == 1.0
    original = tone.to_asset
    monkeypatch.setattr(tone, 'to_asset', lambda: pytest.fail('unchanged frame copied an asset'))
    for _ in range(120):
        assert monitor.update(owner) is None
    monkeypatch.setattr(tone, 'to_asset', original)
    tone.set_param('exposure', .25)
    assert monitor.update(owner)['tonemapping_exposure'] == .25
    owner.pipeline.msaa_samples = MSAASamples.OFF
    assert monitor.update(owner)['msaa_samples'] == 1
    owner.pipeline.msaa_samples = MSAASamples.X4
    assert monitor.update(owner)['msaa_samples'] == 4


def test_slot_disable_replacement_clone_and_scene_without_stack():
    tone = effect(exposure=.5)
    owner = stack(tone)
    monitor = state()
    assert monitor.update(owner)['tonemapping_exposure'] == .5
    owner.effect_slots[0].enabled = False
    assert monitor.update(owner)['tonemapping_mode'] == 0
    clone = tone.clone()
    clone.set_param('exposure', .125)
    owner.effect_slots[0] = EffectSlot(stage_id='final', effect=clone)
    assert monitor.update(owner)['tonemapping_exposure'] == .125
    assert monitor.update(None)['tonemapping_mode'] == 0
    assert monitor.update(None) is None
    assert monitor.update(owner)['tonemapping_exposure'] == .125


def test_live_group_child_revision_and_document_replacement(monkeypatch):
    from infernux.core.assets import AssetManager
    child = effect('bloom', intensity=.8, threshold=1.0)
    group = EditableRenderEffectGroup(RenderEffectGroupAsset((
        RenderEffectGroupEntry('bloom', EffectAssetReference('a'*32), overrides={'intensity': .25}),)))
    monkeypatch.setattr(AssetManager, 'load_by_guid', lambda guid, asset_type: child)
    monitor, owner = state(), stack(group)
    assert monitor.update(owner)['bloom_intensity'] == .25
    child.set_param('threshold', 1.5)
    assert monitor.update(owner)['bloom_threshold'] == 1.5
    assert group.deserialize_document(RenderEffectGroupAsset((
        RenderEffectGroupEntry('bloom', EffectAssetReference('a'*32), overrides={'intensity': .6}),)))
    assert monitor.update(owner)['bloom_intensity'] == .6
    assert group.deserialize_document(RenderEffectGroupAsset())
    assert not monitor.update(owner)['bloom_enabled']


def test_group_cycles_and_missing_assets_are_explicit_errors(monkeypatch):
    from infernux.core.assets import AssetManager
    group = EditableRenderEffectGroup(RenderEffectGroupAsset((
        RenderEffectGroupEntry('cycle', EffectAssetReference('a'*32)),)))
    monkeypatch.setattr(AssetManager, 'load_by_guid', lambda guid, asset_type: group)
    with pytest.raises(RuntimeError, match='cycle'):
        state().update(stack(group))
    monkeypatch.setattr(AssetManager, 'load_by_guid', lambda guid, asset_type: None)
    with pytest.raises(RuntimeError, match='resolve'):
        state().update(stack(group))


def test_unsupported_runtime_samples_and_stages_fail_explicitly():
    owner = stack(effect())
    owner.pipeline.msaa_samples = MSAASamples.X2
    with pytest.raises(RuntimeError, match='OFF or X4'):
        state().update(owner)
    owner.pipeline.msaa_samples = MSAASamples.X4
    owner.effect_slots[0].stage_id = 'after_sky'
    with pytest.raises(RuntimeError, match='final EffectStage'):
        state().update(owner)
