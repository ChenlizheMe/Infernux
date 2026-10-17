from __future__ import annotations

from types import SimpleNamespace

from infernux.components.builtin.sprite_renderer import SpriteRenderer
from infernux.components.sprite_animator import SpriteAnimator
from infernux.core.anim_state_machine import AnimStateMachine
from infernux.core.asset_ref import AnimStateMachineRef
from infernux.engine.interaction import AssetMutation, AssetMutationKind


def _sprite_renderer(guid: str) -> SpriteRenderer:
    renderer = SpriteRenderer()
    renderer._cpp_component = SimpleNamespace(sprite_guid=guid)
    return renderer


def _record_sprite_refresh(renderer: SpriteRenderer, monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(renderer, "_load_sprite_data", lambda: calls.append("load"))
    monkeypatch.setattr(renderer, "_apply_uv_rect", lambda: calls.append("uv"))
    monkeypatch.setattr(renderer, "_apply_color", lambda: calls.append("color"))
    return calls


def test_reused_old_path_with_new_guid_does_not_reload_sprite(monkeypatch, tmp_path):
    renderer = _sprite_renderer("3" * 32)
    calls = _record_sprite_refresh(renderer, monkeypatch)
    reused_path = tmp_path / "sprite.png"

    renderer._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MODIFIED,
            str(reused_path),
            guid="4" * 32,
        )
    )

    assert calls == []


def test_same_guid_move_then_modify_still_reloads_sprite(monkeypatch, tmp_path):
    guid = "5" * 32
    renderer = _sprite_renderer(guid)
    calls = _record_sprite_refresh(renderer, monkeypatch)
    old_path = tmp_path / "old" / "sprite.png"
    new_path = tmp_path / "new" / "sprite.png"

    renderer._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MOVED,
            str(old_path),
            str(new_path),
            guid=guid,
        )
    )
    renderer._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MODIFIED,
            str(new_path),
            guid=guid,
        )
    )

    assert calls == ["load", "uv", "color", "load", "uv", "color"]


def test_sprite_controller_match_invalidates_reference_cache_and_uses_guid_path(monkeypatch, tmp_path):
    from infernux.components import sprite_animator as animator_module

    controller_guid = "9" * 32
    current_path = tmp_path / "moved" / "controller.animfsm"

    class Database:
        @staticmethod
        def get_path_from_guid(guid):
            return str(current_path) if guid == controller_guid else ""

    monkeypatch.setattr(animator_module, "_get_asset_database", lambda: Database())
    reference = AnimStateMachineRef(guid=controller_guid)
    stale_controller = object()
    reference._cached = stale_controller
    animator = SpriteAnimator()
    animator.controller = reference
    animator._fsm = AnimStateMachine(name="stale")
    reloads: list[str] = []
    monkeypatch.setattr(
        animator,
        "_reload_controller_asset",
        lambda path: reloads.append(path) or True,
    )

    animator._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MOVED,
            str(tmp_path / "old" / "controller.animfsm"),
            str(current_path),
            guid=controller_guid,
        )
    )

    assert reference._cached is None
    assert reloads == [str(current_path)]
