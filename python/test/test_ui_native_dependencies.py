"""Native geometry snapshots and sparse resource dependency invalidation."""
import pytest


@pytest.mark.parametrize("world", [False, True])
def test_static_ui_dependencies_do_not_enumerate_python_geometry(scene, monkeypatch, world):
    from Infernux.lib import Vector3
    from Infernux.ui import UICanvas, UIText
    from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
    import Infernux.ui.ui_render_dispatch as module

    parent = scene.create_game_object("Root")
    canvas = None
    if not world:
        canvas = UICanvas()
        parent.add_py_component(canvas)
    text = UIText()
    obj = scene.create_game_object("Text")
    obj.set_parent(parent)
    obj.add_py_component(text)
    unrelated = scene.create_game_object("Unrelated 3D body")
    canvases = () if world else (canvas,)
    worlds = (text,) if world else ()

    def revision():
        return module.runtime_ui_revision(scene, canvases, 800, 600, world_elements=worlds)

    before = revision()
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Unchanged UI traversed Python geometry or empty material slots")
    monkeypatch.setattr(InxUIScreenComponent, "get_rect", unexpected)
    monkeypatch.setattr(InxUIScreenComponent, "world_ui_matrix", unexpected)
    monkeypatch.setattr(module, "material_visual_revision", unexpected)
    if canvas is not None:
        monkeypatch.setattr(canvas, "_get_elements", unexpected)
    for index in range(100):
        assert revision() == before
        unrelated.transform.position = Vector3(index, 0, 0)
        assert revision() == before
    obj.transform.local_position = Vector3(5, 10, 0)
    assert revision() != before
    for index in range(4):
        before = revision()
        obj.transform.local_position = Vector3(8 + index, 10, 0)
        assert revision() != before  # No render/world-matrix sync between edits.


def test_native_ui_pose_contract_and_expired_handles(scene):
    from Infernux.lib import Vector3
    from Infernux.lib._Infernux import _UITransformDependencies

    parent = scene.create_game_object("Parent")
    obj = scene.create_game_object("UI")
    obj.set_parent(parent)
    screen = _UITransformDependencies([obj], [])
    world = _UITransformDependencies([], [obj])
    before_screen, before_world = screen.poll(), world.poll()
    parent.transform.position = Vector3(2, 3, 4)
    assert screen.poll() == before_screen
    assert world.poll() != before_world
    before_world = world.poll()
    obj.transform.local_scale = Vector3(3, 2, 1)
    assert screen.poll() == before_screen
    assert world.poll() == before_world  # UI deliberately ignores its own Scale.
    obj.layer = 5
    assert screen.poll() == before_screen
    assert world.poll() != before_world
    obj.transform.local_euler_angles = Vector3(20, 0, 0)
    assert screen.poll() == before_screen
    before_world = world.poll()
    obj.transform.local_euler_angles = Vector3(20, 0, 30)
    assert screen.poll() != before_screen
    assert world.poll() != before_world
    scene.destroy_game_object(obj)
    scene.process_pending_destroys()
    with pytest.raises(RuntimeError, match="outlived its Transform"):
        world.poll()


def test_ui_binding_changes_and_material_edits_invalidate_native_snapshot(scene):
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    from Infernux.ui import UIText
    from Infernux.ui.ui_render_dispatch import runtime_ui_revision

    text = UIText()
    scene.create_game_object("Text").add_py_component(text)
    def revision():
        return runtime_ui_revision(scene, (), 800, 600, world_elements=(text,))
    empty = revision()
    material = Material(InxMaterial("UI material", "Unlit"))
    text.material = material
    bound = revision()
    assert bound != empty
    assert revision() == bound
    material.set_color("baseColor", .2, .4, .6, 1.)
    assert revision() != bound
    changed = revision()
    text.material = None
    assert revision() != changed
    empty = revision()
    material.set_color("baseColor", .6, .4, .2, 1.)
    assert revision() == empty  # Detached material is no longer a dependency.


@pytest.mark.parametrize("world", [False, True])
def test_shared_ui_materials_are_polled_once_per_publication(scene, monkeypatch, world):
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    from Infernux.ui import UIButton, UICanvas
    import Infernux.ui.ui_render_dispatch as module

    root = scene.create_game_object("Shared UI materials")
    canvas = None if world else root.add_py_component(UICanvas())
    materials = [Material(InxMaterial(f"UI material {i}", "Unlit")) for i in range(4)]
    elements = []
    for i in range(80):
        owner = scene.create_game_object(f"Button {i}")
        owner.set_parent(root)
        button = owner.add_py_component(UIButton())
        button.material = materials[i % 4]
        button.text_material = materials[i % 4]
        elements.append(button)
    calls = []
    original = module.material_visual_revision

    def poll(element, slot):
        calls.append((element, slot))
        return original(element, slot)

    monkeypatch.setattr(module, "material_visual_revision", poll)

    def revision(expected=4):
        calls.clear()
        result = module.runtime_ui_revision(
            scene, () if world else (canvas,), 800, 600,
            world_elements=tuple(elements) if world else (),
        )
        assert len(calls) == expected
        return result

    before = revision()
    assert revision() == before
    elements[-1].text = "Changed text only"
    assert revision() != before
    before = revision()
    materials[2].set_color("baseColor", .3, .6, .9, 1.)
    assert revision() != before
    before = revision()
    # Even without an element setter, a replaced reference is observed on the
    # next publication. No long-lived representative owns the whole group.
    replacement = Material(InxMaterial("Replacement", "Unlit"))
    type(elements[-1]).material.get_raw(elements[-1])._cached = replacement
    assert revision(5) != before
    elements[-1].material = materials[3]
    assert revision() != before


def test_unresolved_ui_material_aliases_do_not_change_snapshot_when_resolved(scene, monkeypatch):
    from Infernux.core.asset_ref import MaterialRef
    from Infernux.core.assets import AssetManager
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    from Infernux.ui import UIText
    import Infernux.ui.ui_render_dispatch as module

    material = Material(InxMaterial("Published UI material", "Unlit"))
    loaded = []
    def load(guid, **kwargs):
        loaded.append(guid)
        return material
    monkeypatch.setattr(AssetManager, "load_by_guid", load)
    elements = []
    for i in range(4):
        text = scene.create_game_object(f"Alias {i}").add_py_component(UIText())
        text.material = MaterialRef(guid="1234567890abcdef1234567890abcdef")
        elements.append(text)
    def revision():
        return module.runtime_ui_revision(scene, (), 800, 600, world_elements=tuple(elements))
    before = revision()
    assert len(loaded) == 1
    assert revision() == before
    for text in elements:
        assert text.material is material
    assert revision() == before
