"""Authoring queues and registry updates publish one complete current candidate."""

import pytest

from infernux.core.asset_reference_types import AssetReferenceType, AssetTypeRegistry
from infernux.core.assets import AssetManager
from infernux.core.animation_timeline import AnimationTimeline


@pytest.mark.parametrize("replace", [False, True])
def test_failed_alias_registration_preserves_the_complete_registry(replace):
    registry = AssetTypeRegistry()
    first = AssetReferenceType("First", "First", frozenset({".one"}), ("FIRST",), "first", aliases=("occupied",))
    second = AssetReferenceType("Second", "Second", frozenset({".two"}), ("SECOND",), "second", aliases=("original",))
    registry.register(first)
    if replace:
        registry.register(second)
    names = ("First", "Second", "original", "partial", "occupied", "OCCUPIED")
    baseline = registry.values(), tuple(registry.get(name) for name in names)
    candidate = AssetReferenceType(
        "Second", "New Second", frozenset({".new"}), ("SECOND",), "second", aliases=("partial", "OCCUPIED")
    )
    with pytest.raises(ValueError, match="belongs to both"):
        registry.register(candidate, replace=replace)
    assert (registry.values(), tuple(registry.get(name) for name in names)) == baseline
    assert registry.get("New Second") is None


@pytest.mark.parametrize("mode", ["replace", "next_flush", "cancel"])
def test_debounced_asset_save_writes_latest_resource(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(AssetManager, "_scheduled_saves", {})
    path = tmp_path / "Shared.animtimeline"
    older = AnimationTimeline(duration=2.0, file_path=str(path))
    newer = AnimationTimeline(duration=7.0, file_path=str(path))
    delay = 0.0 if mode == "next_flush" else 60.0
    if mode == "callback_then_asset":
        AssetManager.schedule_save(str(path), older.save, debounce_sec=delay)
    else:
        AssetManager.schedule_asset_save("animtimeline", str(path), older, debounce_sec=delay)
    if mode == "next_flush":
        assert not AssetManager.flush_scheduled_saves(str(path))
    if mode == "same":
        older.duration = 7.0
        newer = older
    AssetManager.schedule_asset_save("animtimeline", str(path), newer, debounce_sec=delay)
    if mode == "cancel":
        assert AssetManager.cancel_scheduled_save(str(path))
        assert not AssetManager.flush_scheduled_saves(str(path), force=True)
        assert not path.exists()
        return
    assert AssetManager.flush_scheduled_saves(str(path), force=mode != "next_flush")
    assert AnimationTimeline.load(str(path)).duration == 7.0
    assert not AssetManager.flush_scheduled_saves(str(path), force=True)
