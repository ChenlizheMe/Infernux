from infernux.engine.ui.project_file_ops import copy_path_as_new_asset
from infernux.particle import ParticleGraphAsset


def test_particle_graph_copy_regenerates_only_asset_identity(tmp_path):
    source = tmp_path / "Source.particlegraph"
    destination = tmp_path / "Copied.particlegraph"
    original = ParticleGraphAsset(name="Source")
    original.save(str(source))

    copied_path = copy_path_as_new_asset(str(source), str(destination))
    copied = ParticleGraphAsset.load(str(destination))

    assert copied_path == str(destination.resolve())
    assert copied.stable_id != original.stable_id
    assert copied.name == "Copied"
    assert tuple(emitter.stable_id for emitter in copied.emitters) == tuple(
        emitter.stable_id for emitter in original.emitters
    )
    assert destination.with_suffix(".particlegraph.meta").exists() is False


def test_folder_copy_excludes_meta_and_regenerates_nested_particle_graph(tmp_path):
    source = tmp_path / "Source"
    destination = tmp_path / "Copied"
    source.mkdir()
    graph_path = source / "Smoke.particlegraph"
    graph = ParticleGraphAsset(name="Smoke")
    graph.save(str(graph_path))
    (source / "Smoke.particlegraph.meta").write_text("stale-guid", encoding="utf-8")
    (source / "note.txt").write_text("hello", encoding="utf-8")

    assert copy_path_as_new_asset(str(source), str(destination)) == str(destination.resolve())
    copied = ParticleGraphAsset.load(str(destination / "Smoke.particlegraph"))
    assert copied.stable_id != graph.stable_id
    assert (destination / "Smoke.particlegraph.meta").exists() is False
    assert (destination / "note.txt").read_text(encoding="utf-8") == "hello"
