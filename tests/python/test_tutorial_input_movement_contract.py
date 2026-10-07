"""Execute the published mover and its documented mistakes with native input."""
import math
from pathlib import Path
import re

import pytest

from infernux import Destroy
from infernux.input import KeyCode
from infernux.lib import SceneManager
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline


CHAPTER = Path(__file__).resolve().parents[2] / "docs/learn/gameplay-input-movement.md"


@pytest.fixture(scope="module")
def movers():
    source = re.search(r"```python\n(.*?)```", CHAPTER.read_text(encoding="utf-8"), re.S).group(1)
    normalization_start = source.index("        length_squared =")
    normalization_end = source.index("        frame_distance =")
    variants = {
        "published": source,
        "pressed_once": source.replace("Input.get_key(", "Input.get_key_down("),
        "per_frame": source.replace("self.speed * delta_time", "self.speed"),
        "unnormalized": source[:normalization_start] + source[normalization_end:],
    }
    classes = {}
    for name, code in variants.items():
        namespace = {"__name__": "tutorial_keyboard_movement_contract." + name}
        exec(compile(code, str(CHAPTER), "exec"), namespace)
        classes[name] = namespace["KeyboardMover"]
    return classes


def travel(scene, engine, component_type, steps, keys):
    owner = scene.create_game_object("TutorialMover")
    owner.add_component(component_type)
    manager = SceneManager.instance()
    pipeline = RenderStackPipeline()
    engine.set_render_pipeline(pipeline)
    manager.play()
    try:
        # Scene publication discards its first frame delta. Consume that
        # loading boundary before measuring elapsed movement or input edges.
        engine.tick(0.0)
        for key in keys:
            engine.queue_synthetic_key_input(key, True)
        for duration in steps:
            engine.tick(duration)
        position = owner.transform.position
        return math.hypot(position.x, position.z)
    finally:
        for key in keys:
            engine.queue_synthetic_key_input(key, False)
        engine.tick(0.0)
        manager.stop()
        Destroy(owner)
        engine.set_render_pipeline(None)


@pytest.mark.parametrize("steps", [(0.025,) * 4, (0.05,) * 2])
def test_published_mover_covers_the_same_distance_for_the_same_elapsed_time(scene, engine, movers, steps):
    assert travel(scene, engine, movers["published"], steps, [KeyCode.W]) == pytest.approx(0.4, abs=1e-5)


@pytest.mark.parametrize("variant,expected", [("pressed_once", 0.1), ("per_frame", 16.0)])
def test_documented_time_and_input_mistakes_have_the_explained_effect(scene, engine, movers, variant, expected):
    assert travel(scene, engine, movers[variant], (0.025,) * 4, [KeyCode.W]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("variant,expected", [("published", 0.4), ("unnormalized", 0.4 * math.sqrt(2))])
def test_diagonal_normalization_prevents_the_documented_speed_boost(scene, engine, movers, variant, expected):
    assert travel(scene, engine, movers[variant], (0.025,) * 4, [KeyCode.W, KeyCode.D]) == pytest.approx(expected, abs=1e-5)


def test_legacy_keycode_spelling_is_not_a_public_constant():
    with pytest.raises(AttributeError):
        getattr(KeyCode, "UpArrow")
    assert isinstance(KeyCode.UP_ARROW, int)
