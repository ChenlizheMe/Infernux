"""Sprite fields are renderer state, never shared material authoring edits."""
from pathlib import Path
import copy

import pytest
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.core.asset_types import SpriteFrame, TextureImportSettings, TextureType, write_texture_import_settings
from infernux.core.material import Material
from infernux.engine.ui import project_file_ops
from model_test_support import remove_model_test_folder


@pytest.fixture
def sprite_assets(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    frames = [SpriteFrame(stable_id=str(index) * 32, name=f"Frame{index}",
                          x=(index - 1) * 4, y=0, w=4, h=4) for index in (1, 2)]
    textures = []
    for index, color in enumerate(((255, 255, 255, 255), (128, 255, 128, 255))):
        path = folder / f"Sheet{index}.png"
        Image.new("RGBA", (8, 4), color).save(path)
        result = AssetManager.import_asset(str(path), database=database)
        assert result, result.error
        settings = TextureImportSettings(texture_type=TextureType.SPRITE, sprite_frames=frames)
        assert write_texture_import_settings(str(path), settings)
        result = AssetManager.reimport_asset(str(path), database=database)
        assert result, result.error
        textures.append(result.guid)
    created, error = project_file_ops.create_material(str(folder), "Shared", database)
    assert created, error
    material_path = folder / "Shared.mat"
    material = Material.load(str(material_path))
    material.frag_shader_name = "Sprite Unlit"
    material.set_color("baseColor", .25, .5, .75, 1.)
    material.set_texture("texSampler", textures[0])
    material.flush()
    try:
        yield material, material_path, textures, frames
    finally:
        # Release live renderers before invalidating their test assets.
        for owner in list(scene.get_root_objects()):
            scene._remove_game_object_immediately(owner)
        Material.flush_all_pending()
        remove_model_test_folder(database, folder)


def _sprite(scene, name, material):
    renderer = scene.create_game_object(name).add_component("SpriteRenderer")
    renderer.material = material
    return renderer


def test_sprite_instances_keep_shared_document_and_parameters_independent(scene, sprite_assets):
    material, path, textures, frames = sprite_assets
    before = path.read_bytes()
    authored = copy.deepcopy(material.serialize_document())
    left = _sprite(scene, "Left", material)
    right = _sprite(scene, "Right", material)
    for renderer, texture, frame, color, flip in (
        (left, textures[0], frames[0], (1., 0., 0., 1.), False),
        (right, textures[1], frames[1], (0., 0., 1., 1.), True),
    ):
        renderer.sprite = texture
        renderer.frame_id = frame.stable_id
        renderer.sprite_color = color
        renderer.flip_x = flip
        renderer.sync_visual()
    Material.flush_all_pending()
    assert path.read_bytes() == before
    assert material.serialize_document() == authored
    assert left.material is material and right.material is material
    assert left.get_parameter("baseColor") == pytest.approx((1, 0, 0, 1))
    assert right.get_parameter("baseColor") == pytest.approx((0, 0, 1, 1))
    assert left.get_parameter("uvRect") == pytest.approx((0, 1, .5, -1))
    assert right.get_parameter("uvRect") == pytest.approx((1, 1, -.5, -1))
    assert left.get_parameter("texSampler") == textures[0]
    assert right.get_parameter("texSampler") == textures[1]
    for renderer in (left, right):
        document = renderer.serialize_document()
        assert not document.get("parameterOverrides")
        assert document["materials"] == [material.guid]


def test_sprite_document_restore_rebuilds_derived_values_without_authoring_material(scene, sprite_assets):
    material, path, textures, frames = sprite_assets
    before = path.read_bytes()
    renderer = _sprite(scene, "Sprite", material)
    renderer.sprite = textures[1]
    renderer.frame_id = frames[1].stable_id
    renderer.sprite_color = (0.1, 0.2, 0.3, 1.)
    renderer.sync_visual()
    document = renderer.serialize_document()
    assert renderer.deserialize_document(copy.deepcopy(document))
    renderer.sync_visual()
    assert renderer.get_parameter("baseColor") == pytest.approx((.1, .2, .3, 1))
    assert renderer.get_parameter("texSampler") == textures[1]
    assert renderer.get_parameter("uvRect") == pytest.approx((.5, 1, .5, -1))
    Material.flush_all_pending()
    assert path.read_bytes() == before


def test_clearing_one_sprite_texture_does_not_clear_other_sprite_or_shared_asset(scene, sprite_assets):
    material, path, textures, _ = sprite_assets
    before = path.read_bytes()
    left = _sprite(scene, "Left", material)
    right = _sprite(scene, "Right", material)
    left.sprite = textures[0]
    right.sprite = textures[1]
    left.sprite = ""
    assert left.get_parameter("texSampler") == ""
    assert right.get_parameter("texSampler") == textures[1]
    Material.flush_all_pending()
    assert path.read_bytes() == before


def test_sprite_undo_redo_restores_parameters_and_empty_texture(scene, sprite_assets):
    from infernux.engine.undo import GenericComponentCommand

    material, path, textures, frames = sprite_assets
    before = path.read_bytes()
    renderer = _sprite(scene, "UndoSprite", material)
    empty = renderer.serialize_document()
    renderer.sprite = textures[1]
    renderer.frame_id = frames[1].stable_id
    renderer.flip_y = True
    renderer.sprite_color = (.4, .5, .6, 1.)
    renderer.sync_visual()
    populated = renderer.serialize_document()
    command = GenericComponentCommand(renderer, empty, populated)
    for _ in range(3):
        command.undo()
        assert renderer.sprite == ""
        assert renderer.get_parameter("texSampler") == ""
        assert renderer.get_parameter("baseColor") == pytest.approx((1, 1, 1, 1))
        command.redo()
        assert renderer.get_parameter("texSampler") == textures[1]
        assert renderer.get_parameter("baseColor") == pytest.approx((.4, .5, .6, 1))
        assert renderer.get_parameter("uvRect") == pytest.approx((.5, 0, .5, 1))
    Material.flush_all_pending()
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["clone", "prefab"])
def test_sprite_copy_rebuilds_instance_parameters(scene, sprite_assets, operation):
    from infernux.engine.component_restore import clone_game_object_transactionally
    from infernux.engine.prefab_manager import save_prefab, instantiate_prefab

    material, path, textures, frames = sprite_assets
    before = path.read_bytes()
    original = _sprite(scene, "Source", material)
    original.sprite = textures[1]
    original.frame_id = frames[1].stable_id
    original.sprite_color = (.2, .4, .6, 1.)
    original.sync_visual()
    if operation == "clone":
        copied = clone_game_object_transactionally(scene, original.game_object)
    else:
        prefab_path = path.parent / "Sprite.prefab"
        assert save_prefab(original.game_object, str(prefab_path))
        copied = instantiate_prefab(file_path=str(prefab_path), guid="sprite-test-prefab", scene=scene)
    renderer = copied.get_component("SpriteRenderer")
    assert renderer.component_id != original.component_id
    assert renderer.game_object.id == copied.id
    assert original.game_object.name == "Source"
    assert renderer.material is material
    assert renderer.get_parameter("texSampler") == textures[1]
    assert renderer.get_parameter("uvRect") == pytest.approx((.5, 1, .5, -1))
    assert renderer.get_parameter("baseColor") == pytest.approx((.2, .4, .6, 1))
    renderer.sprite_color = (1., 0., 0., 1.)
    renderer.sync_visual()
    assert original.get_parameter("baseColor") == pytest.approx((.2, .4, .6, 1))
    Material.flush_all_pending()
    assert path.read_bytes() == before


def test_sprite_texture_reimport_refreshes_only_its_instance(scene, sprite_assets):
    from infernux.engine.interaction import AssetMutation, AssetMutationKind

    material, path, textures, frames = sprite_assets
    before = path.read_bytes()
    left = _sprite(scene, "Left", material)
    right = _sprite(scene, "Right", material)
    left.sprite = textures[0]
    right.sprite = textures[1]
    texture_path = path.parent / "Sheet0.png"
    settings = TextureImportSettings(texture_type=TextureType.SPRITE, sprite_frames=[
        SpriteFrame(stable_id=frames[0].stable_id, name="Cropped", x=2, y=1, w=2, h=2),
        frames[1],
    ])
    assert write_texture_import_settings(str(texture_path), settings)
    result = AssetManager.reimport_asset(str(texture_path), database=AssetManager._asset_database)
    assert result, result.error
    change = AssetMutation(AssetMutationKind.MODIFIED, str(texture_path), guid=textures[0])
    # Exercise the callback used by the mutation service after the native
    # asset database has committed, including an unrelated subscriber.
    left._on_asset_changed(change)
    right._on_asset_changed(change)
    assert left.get_parameter("uvRect") == pytest.approx((.25, .75, .25, -.5))
    assert right.get_parameter("uvRect") == pytest.approx((0, 1, .5, -1))
    Material.flush_all_pending()
    assert path.read_bytes() == before


def test_sprite_shader_refresh_keeps_other_parameter_owners(scene, sprite_assets):
    material, _, textures, _ = sprite_assets
    renderer = _sprite(scene, "Sprite", material)
    renderer.sprite = textures[0]
    renderer.set_parameter("baseColor", (0., 1., 0., 1.), owner="gameplay")
    renderer.sync_visual()
    assert renderer.get_parameter("baseColor") == pytest.approx((0, 1, 0, 1))
    material.set_color("baseColor", .5, .5, .5, 1.)
    material.flush()
    renderer.sync_visual()
    assert renderer.get_parameter("baseColor", owner="gameplay") == pytest.approx((0, 1, 0, 1))
    assert renderer.get_parameter("baseColor", owner="sprite-renderer") == pytest.approx((1, 1, 1, 1))
    assert renderer.get_parameter("texSampler") == textures[0]
