from __future__ import annotations

from types import SimpleNamespace

from Infernux.components.builtin.sprite_renderer import SpriteRenderer
from Infernux.components.spirit_animator import SpiritAnimator
from Infernux.core.anim_state_machine import AnimState, AnimStateMachine
from Infernux.core.asset_ref import AnimStateMachineRef
from Infernux.engine.interaction import AssetMutation, AssetMutationKind


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


def test_sprite_hot_reload_matches_asset_mutation_guid(monkeypatch, tmp_path):
    guid = "1" * 32
    renderer = _sprite_renderer(guid)
    calls = _record_sprite_refresh(renderer, monkeypatch)

    renderer._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MODIFIED,
            str(tmp_path / "arbitrary-new-location.png"),
            guid=guid,
        )
    )

    assert calls == ["load", "uv", "color"]


def test_empty_guid_mutation_never_reloads_registered_sprite(monkeypatch, tmp_path):
    renderer = _sprite_renderer("2" * 32)
    calls = _record_sprite_refresh(renderer, monkeypatch)

    renderer._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MODIFIED,
            str(tmp_path / "sprite.png"),
        )
    )

    assert calls == []


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


def test_spirit_clip_reload_resolves_current_path_only_after_guid_match(
    monkeypatch, tmp_path
):
    from Infernux.components import spirit_animator as animator_module

    clip_guid = "6" * 32
    current_path = tmp_path / "moved" / "walk.animclip2d"
    resolved: list[str] = []

    class Database:
        @staticmethod
        def get_path_from_guid(guid):
            resolved.append(guid)
            return str(current_path) if guid == clip_guid else ""

    monkeypatch.setattr(animator_module, "_get_asset_database", lambda: Database())
    animator = SpiritAnimator()
    animator._fsm = AnimStateMachine(
        states=[AnimState(name="Walk", clip_guid=clip_guid)],
        default_state="Walk",
    )
    reloads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        animator,
        "_reload_clip_asset",
        lambda guid, path: reloads.append((guid, path)) or True,
    )

    animator._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MODIFIED,
            str(tmp_path / "old" / "walk.animclip2d"),
            guid=clip_guid,
        )
    )

    assert resolved == [clip_guid]
    assert reloads == [(clip_guid, str(current_path))]


def test_spirit_ignores_empty_and_wrong_guid_before_database_resolution(
    monkeypatch, tmp_path
):
    from Infernux.components import spirit_animator as animator_module

    clip_guid = "7" * 32

    class Database:
        @staticmethod
        def get_path_from_guid(_guid):
            raise AssertionError("unmatched mutations must not resolve a path")

    monkeypatch.setattr(animator_module, "_get_asset_database", lambda: Database())
    animator = SpiritAnimator()
    animator._fsm = AnimStateMachine(
        states=[AnimState(name="Walk", clip_guid=clip_guid)],
        default_state="Walk",
    )
    old_path = tmp_path / "walk.animclip2d"

    animator._on_asset_changed(
        AssetMutation(AssetMutationKind.MODIFIED, str(old_path))
    )
    animator._on_asset_changed(
        AssetMutation(
            AssetMutationKind.MODIFIED,
            str(old_path),
            guid="8" * 32,
        )
    )


def test_spirit_controller_match_invalidates_reference_cache_and_uses_guid_path(
    monkeypatch, tmp_path
):
    from Infernux.components import spirit_animator as animator_module

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
    animator = SpiritAnimator()
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
