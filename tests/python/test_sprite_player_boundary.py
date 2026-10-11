"""Sprite runtime lifecycles never load authoring-only asset subscriptions."""
import builtins

import pytest

from infernux.application import Application
from infernux.components.builtin.sprite_renderer import SpriteRenderer
from infernux.components.spirit_animator import SpiritAnimator


@pytest.fixture
def player_without_authoring(monkeypatch):
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    original = builtins.__import__

    def reject_authoring(name, *args, **kwargs):
        if name == "infernux.engine.interaction" or name.startswith("infernux.engine.interaction."):
            raise AssertionError("Player attempted to import authoring asset mutation services")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_authoring)


def test_sprite_native_binding_is_valid_without_editor_asset_services(scene, player_without_authoring):
    owner = scene.create_game_object("Player Sprite")
    renderer = owner.add_component(SpriteRenderer)
    assert renderer.component_id > 0
    assert renderer.material is not None
    assert renderer.__dict__.get("_asset_mutation_service") is None
    renderer.sync_visual()
    renderer._invalidate_native_binding()


def test_spirit_animation_lifecycle_does_not_attempt_authoring_imports(scene, player_without_authoring):
    owner = scene.create_game_object("Player Animation")
    owner.add_component(SpriteRenderer)
    animator = owner.add_component(SpiritAnimator)
    animator.awake()
    animator.start()
    assert isinstance(animator._sprite_renderer, SpriteRenderer)
    assert getattr(animator, "_asset_mutation_service", None) is None
    animator.on_destroy()
