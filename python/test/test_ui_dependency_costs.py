"""UI dependency checks must stay cheap without weakening invalidation."""
from types import SimpleNamespace

import pytest


def test_canvas_ancestor_snapshot_reuses_topology_and_tracks_reparent(scene, monkeypatch):
    from Infernux.lib import GameObject, Vector3
    from Infernux.ui import UICanvas, UIText

    root = scene.create_game_object("Canvas")
    canvas = UICanvas()
    root.add_py_component(canvas)
    parent = root
    for index in range(12):
        child = scene.create_game_object(f"Group {index}")
        child.set_parent(parent)
        parent = child
    text = UIText()
    parent.add_py_component(text)

    original = GameObject.get_py_components
    reads = []

    def read_components(obj):
        reads.append(obj.id)
        return original(obj)

    monkeypatch.setattr(GameObject, "get_py_components", read_components)
    # Adding the component invalidated the authoring-time snapshot.
    assert text.get_canvas() is canvas
    reads.clear()
    for _ in range(100):
        assert text.get_canvas() is canvas
    assert reads == []
    root.transform.position = Vector3(10, 20, 0)
    assert text.get_canvas() is canvas
    assert reads == []

    parent.set_parent(None)
    assert text.get_canvas() is None
    assert reads
    reads.clear()
    for _ in range(100):
        assert text.get_canvas() is None
    assert reads == []

    other_canvas = UICanvas()
    parent.add_py_component(other_canvas)
    assert text.get_canvas() is other_canvas
    parent.remove_py_component(other_canvas)
    assert text.get_canvas() is None
    replacement = scene.create_game_object("Replacement Canvas")
    replacement_canvas = UICanvas()
    replacement.add_py_component(replacement_canvas)
    # A different owner must not inherit the cached ancestry of this wrapper.
    unbound = UIText()
    assert unbound.get_canvas() is None
    replacement.add_py_component(unbound)
    assert unbound.get_canvas() is replacement_canvas


@pytest.mark.parametrize("slot", ["material", "text_material"])
def test_material_dependency_check_does_not_resolve_draw_data(slot):
    from Infernux.ui.ui_render_dispatch import material_visual_revision

    class Native:
        guid = "same-guid"
        _texture_assets_pending = False
        version = 2
        target = None

        def get_version(self):
            return self.version

        def _get_render_texture(self, name):
            assert name == "texSampler"
            return self.target

        def has_property(self, name):
            raise AssertionError("A revision check read material draw data")

        def get_color(self, name):
            raise AssertionError("A revision check read a color")

        def get_texture(self, name):
            raise AssertionError("A revision check resolved a texture path")

    native = Native()
    element = SimpleNamespace(**{slot: SimpleNamespace(native=native)})
    revision = material_visual_revision(element, slot)
    for _ in range(100):
        assert material_visual_revision(element, slot) == revision
    native.version += 1
    assert material_visual_revision(element, slot) != revision
    revision = material_visual_revision(element, slot)
    native.target = SimpleNamespace(revision=10)
    assert material_visual_revision(element, slot) != revision
    revision = material_visual_revision(element, slot)
    native.target.revision += 1
    assert material_visual_revision(element, slot) != revision
    replacement = Native()
    replacement.version = native.version
    replacement.target = native.target
    revision = material_visual_revision(element, slot)
    setattr(element, slot, SimpleNamespace(native=replacement))
    assert material_visual_revision(element, slot) != revision


def test_private_ui_bookkeeping_does_not_read_previous_value_or_dirty_layout():
    from Infernux.ui import UIText
    from Infernux.ui.ui_render_revision import get_runtime_ui_revision

    class Text(UIText):
        @property
        def _bookkeeping(self):
            raise AssertionError("Internal write unexpectedly read the old value")

        @_bookkeeping.setter
        def _bookkeeping(self, value):
            object.__setattr__(self, "_stored_bookkeeping", value)

    text = Text()
    before = get_runtime_ui_revision()
    text._bookkeeping = 42
    assert text._stored_bookkeeping == 42
    assert get_runtime_ui_revision() == before
    text._text_layout_key = ("cached",)
    text.text = "new visible text"
    assert text._text_layout_key is None
    assert get_runtime_ui_revision() > before


def test_real_native_material_reference_keeps_identity_and_publishes_edits(scene):
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    from Infernux.ui import UIImage
    from Infernux.ui.ui_render_dispatch import material_visual_revision, material_visual_state

    native = InxMaterial("UI dependency material", "Unlit")
    material = Material(native)
    material.set_color("baseColor", 1., .5, .25, 1.)
    image = UIImage()
    scene.create_game_object("Image").add_py_component(image)
    image.material = material
    before = material_visual_revision(image)
    for _ in range(100):
        assert material_visual_revision(image) == before
    material.set_color("baseColor", .25, .5, 1., 1.)
    assert material_visual_revision(image) != before
    assert material_visual_state(image)['color'] == pytest.approx((.25, .5, 1., 1.))
