"""Material collection edits replace the complete native renderer state."""
from pathlib import Path

import pytest
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.engine.ui import project_file_ops
from infernux.lib import AssetDependencyGraph
from model_test_support import remove_model_test_folder


RENDERERS = ["MeshRenderer", "LineRenderer", "SkinnedMeshRenderer"]
ENTRIES = ["materials", "sharedMaterials", "set_materials", "set_shared_materials", "SetMaterials", "SetSharedMaterials"]


def replace(renderer, entry, values):
    if entry in ("materials", "sharedMaterials"):
        setattr(renderer, entry, values)
    else:
        getattr(renderer, entry)(values)


@pytest.mark.parametrize("kind", RENDERERS)
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("count", [0, 1])
def test_collection_replacement_removes_tail_slots(scene, kind, entry, count):
    renderer = scene.create_game_object("Collection").add_component(kind)
    originals = [Material.create_unlit(str(i)) for i in range(3)]
    for index, material in enumerate(originals):
        renderer.set_material(index, material)
    replacement = Material.create_unlit("Replacement")
    replace(renderer, entry, [replacement] * count)
    assert renderer.material_count == count
    assert len(renderer.serialize_document()["materials"]) == count
    if count:
        assert renderer.get_material(0).native is replacement.native
    assert renderer.get_material(2) is None


@pytest.mark.parametrize("kind", RENDERERS)
@pytest.mark.parametrize("entry", ENTRIES + ["native"])
def test_invalid_collection_never_publishes_prefix(scene, kind, entry):
    renderer = scene.create_game_object("Invalid collection").add_component(kind)
    previous = Material.create_unlit("Previous")
    renderer.material = previous
    before = renderer.serialize_document()
    candidate = Material.create_unlit("Candidate")
    with pytest.raises(TypeError):
        if entry == "native":
            renderer._cpp_component.set_materials([candidate.native, object()])
        else:
            replace(renderer, entry, [candidate, object()])
    assert renderer.serialize_document() == before
    assert renderer.material.native is previous.native


@pytest.mark.parametrize("kind", RENDERERS)
def test_unavailable_material_guid_is_preserved_and_empty_entries_clear_instances(scene, kind):
    renderer = scene.create_game_object("Missing material").add_component(kind)
    missing = "b" * 32
    renderer.materials = [missing]
    assert renderer.get_material_guids() == [missing]
    assert renderer.serialize_document()["materials"] == [missing]
    transient = Material.create_unlit("To clear")
    renderer.materials = [transient, transient]
    renderer.materials = [None, ""]
    assert renderer.material_count == 2 and renderer.materials == [None, None]


@pytest.fixture
def authored_material(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    created, error = project_file_ops.create_material(str(folder), "Author", database)
    assert created, error
    material = Material.load(str(folder / "Author.mat"))
    try:
        yield material
    finally:
        for owner in list(scene.get_root_objects()):
            scene._remove_game_object_immediately(owner)
        Material.flush_all_pending()
        remove_model_test_folder(database, folder)


@pytest.mark.parametrize("kind", RENDERERS)
@pytest.mark.parametrize("native", [False, True])
def test_mixed_materials_keep_identity_and_survive_document_reload(scene, kind, native, authored_material):
    renderer = scene.create_game_object("Mixed collection").add_component(kind)
    transient = Material.create_unlit("Transient")
    wrapped = Material.create_unlit("Wrapped")
    values = [authored_material.guid, transient.native, None, wrapped]
    if native:
        renderer._cpp_component.set_materials([authored_material.guid, transient.native, None, wrapped.native])
    else:
        renderer.materials = values
    assert renderer.material_count == 4
    assert renderer.get_material(0).guid == authored_material.guid
    assert renderer.get_material(1).native is transient.native
    assert renderer.get_material(2) is None
    assert renderer.get_material(3).native is wrapped.native
    document = renderer.serialize_document()
    assert renderer.deserialize_document(document)
    assert renderer.serialize_document() == document


@pytest.mark.parametrize("kind", RENDERERS)
def test_truncated_slots_release_parameters_and_material_dependencies(engine, scene, kind, authored_material):
    authored_material.set_texture("texSampler", "white")
    authored_material.flush()
    database = engine.get_asset_database()
    folder = Path(database.get_path_from_guid(authored_material.guid)).parent
    texture_guids = []
    for index in range(2):
        path = folder / f"Parameter{index}.png"
        Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(path)
        imported = AssetManager.import_asset(str(path), database=database)
        assert imported, imported.error
        texture_guids.append(imported.guid)
    renderer = scene.create_game_object("Truncate state").add_component(kind)
    graph = AssetDependencyGraph.instance()
    before = set(graph.get_dependents(authored_material.guid))
    material = Material.create_unlit("Retained")
    renderer.set_material(0, material)
    renderer.set_material(2, authored_material)
    renderer.set_parameter("baseColor", (.25, .5, .75, 1.), material_slot=0, persistent=True)
    owned = set(graph.get_dependents(authored_material.guid)) - before
    assert owned
    renderer.set_parameter("baseColor", (1., 0., 0., 1.), material_slot=2, persistent=True)
    renderer.set_parameter("baseColor", (0., 1., 0., 1.), material_slot=2, owner="test")
    renderer.set_parameter("texSampler", texture_guids[0], material_slot=2, persistent=True)
    renderer.set_parameter("texSampler", texture_guids[1], material_slot=2, owner="test")
    for guid in texture_guids:
        assert owned <= set(graph.get_dependents(guid))
    renderer.materials = [material]
    assert renderer.get_parameter("baseColor", material_slot=0, persistent_only=True) == pytest.approx((.25, .5, .75, 1.))
    assert not (owned & set(graph.get_dependents(authored_material.guid)))
    for guid in texture_guids:
        assert not (owned & set(graph.get_dependents(guid)))
    renderer.set_material_slot_count(3)
    assert renderer.get_parameter("baseColor", material_slot=2, persistent_only=True) is None
    assert renderer.get_parameter("baseColor", material_slot=2, owner="test") is None
