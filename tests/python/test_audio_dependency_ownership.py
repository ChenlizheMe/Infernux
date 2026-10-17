"""A source owns one dependency edge per clip, not one edge per track."""
import json
import wave

import pytest

from infernux.lib import AssetDependencyGraph, AudioClip
from infernux.core.assets import AssetManager


def write_wave(path, frames=4410):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0\0" * frames)


@pytest.fixture
def audio_assets(engine, scene, tmp_path):
    database = engine.get_asset_database()
    paths = [tmp_path / name for name in ("共享.wav", "replacement.wav")]
    for path in paths:
        write_wave(path)
    guids = [database.import_asset(str(path)).guid for path in paths]
    assert all(guids)
    owner = scene.create_game_object("Audio dependency owner")
    source = owner.add_component("AudioSource")._require_cpp_component()
    source.play_on_awake = False
    source.track_count = 3
    graph = AssetDependencyGraph.instance()
    previous = set(graph.get_dependents(guids[0]))
    for index in range(3):
        source.set_track_clip_by_guid(index, guids[0])
    dependents = set(graph.get_dependents(guids[0])) - previous
    assert len(dependents) == 1
    identity = dependents.pop()
    try:
        yield source, graph, identity, guids, paths, database
    finally:
        scene._remove_game_object_immediately(owner)
        assert not graph.get_dependencies(identity)
        for path in paths:
            database.delete_asset(str(path))


def test_shrinking_all_tracks_of_one_clip_preserves_other_clip_edges(audio_assets):
    source, graph, identity, guids, _paths, _database = audio_assets
    source.set_track_clip_by_guid(0, guids[1])
    source.track_count = 1
    assert set(graph.get_dependencies(identity)) == {guids[1]}
    source.track_count = 3
    assert source.get_track_clip_guid(1) == source.get_track_clip_guid(2) == ""
    assert set(graph.get_dependencies(identity)) == {guids[1]}


def test_retained_shared_track_still_receives_asset_reimport_notifications(audio_assets):
    source, graph, identity, guids, paths, database = audio_assets
    source.loop = True
    source.play(0)
    assert source.is_track_playing(0)
    source.set_track_clip_by_guid(1, "")
    source.set_track_clip_by_guid(2, "")
    write_wave(paths[0], frames=8820)
    assert AssetManager.reimport_asset(str(paths[0]), database=database)
    assert not source.is_track_playing(0)
    assert source.get_track_clip(0).duration == pytest.approx(.2)
    assert graph.has_dependency(identity, guids[0])


def test_track_edit_workflow_keeps_shared_clip_edges_until_the_last_reference(audio_assets):
    source, graph, identity, guids, paths, _database = audio_assets
    first, second = guids
    source.set_track_clip_by_guid(1, first)
    assert set(graph.get_dependencies(identity)) == {first}
    source.set_track_clip_by_guid(1, second)
    assert set(graph.get_dependencies(identity)) == {first, second}
    transient = AudioClip()
    assert transient.load_from_file(str(paths[1]))
    source.set_track_clip(2, transient)
    assert graph.has_dependency(identity, first) and identity in graph.get_dependents(first)
    assert source.get_track_clip_guid(0) == first
    source.set_track_clip_by_guid(0, "")
    assert not graph.has_dependency(identity, first) and identity not in graph.get_dependents(first)
    assert graph.has_dependency(identity, second)
    source.track_count = 1
    assert not graph.get_dependencies(identity)
    source.track_count = 2
    source.set_track_clip_by_guid(0, first)
    source.set_track_clip_by_guid(1, first)
    assert graph.has_dependency(identity, first)
    document = json.loads(source.serialize())
    document["tracks"][1].pop("clip_guid")
    assert source.deserialize(json.dumps(document))
    assert source.get_track_clip_guid(0) == first
    assert graph.has_dependency(identity, first) and identity in graph.get_dependents(first)
    document["tracks"][0].pop("clip_guid")
    assert source.deserialize(json.dumps(document))
    assert not graph.has_dependency(identity, first) and identity not in graph.get_dependents(first)
