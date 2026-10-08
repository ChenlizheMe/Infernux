"""Tutorial audio loading keeps frozen import settings without source sidecars."""
from pathlib import Path
import os
import subprocess
import sys


def test_public_audio_load_preserves_frozen_player_import_settings(tmp_path):
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, '_INFERNUX_PLAYER_MODE': '1'},
        capture_output=True, text=True, encoding='utf-8', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_audio_import_setting_changes_reach_existing_shared_tracks(engine, scene, tmp_path):
    import shutil
    from infernux.core.assets import AssetManager
    from infernux.core.asset_types import read_audio_import_settings, write_audio_import_settings

    path = tmp_path / '共享音乐.mp3'
    fixture = Path(__file__).resolve().parents[1] / 'native/fixtures/audio/stream_probe.mp3'
    shutil.copyfile(fixture, path)
    database = engine.get_asset_database()
    imported = database.import_asset(str(path))
    assert imported, imported.error
    owner = scene.create_game_object('Audio import settings')
    source = owner.add_component('AudioSource')._require_cpp_component()
    source.play_on_awake = False
    source.loop = True
    source.track_count = 2
    for index in range(2):
        source.set_track_clip_by_guid(index, imported.guid)
    try:
        for mono, streaming in ((True, True), (False, False), (True, False), (False, True)):
            source.play(0)
            source.play(1)
            assert all(source.is_track_playing(index) for index in range(2))
            settings = read_audio_import_settings(str(path))
            settings.force_mono = mono
            settings.load_type = 'streaming' if streaming else 'decompress_on_load'
            assert write_audio_import_settings(str(path), settings)
            reimported = AssetManager.reimport_asset(str(path), database=database)
            assert reimported, reimported.error
            assert reimported.guid == imported.guid
            for index in range(2):
                assert not source.is_track_playing(index)
                assert source.get_track_clip_guid(index) == imported.guid
                clip = source.get_track_clip(index)
                assert clip.channels == (1 if mono else 2)
                assert clip.is_streaming == streaming
    finally:
        source.stop_all()
        scene._remove_game_object_immediately(owner)
        database.delete_asset(str(path))


def exercise_public_player_audio(root):
    import json
    import shutil
    import struct
    import wave

    import infernux as inx
    from infernux.engine.engine import Engine
    from infernux.engine import project_context
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode

    # A frozen catalog fixture exercises the runtime reader independently of
    # cooking. Encoded-byte packaging has its own real import/export test.
    player = root / '新设备' / 'Game_Data'
    artifacts = player / 'Library/Artifacts/Audio'
    artifacts.mkdir(parents=True)
    records, catalog, cases = [], [], []
    fixtures = Path(__file__).resolve().parents[1] / 'native/fixtures/audio'
    for extension in ('wav', 'mp3', 'ogg', 'flac'):
        for mono in (False, True):
            for streaming in (False, True):
                guid = f'{len(cases) + 1:032x}'
                authored = f'Assets/Audio/music-{mono}-{streaming}.{extension}'
                relative = f'Library/Artifacts/Audio/{guid}.{extension}'
                artifact_id = f'audio:{guid}'
                target = player / relative
                if extension == 'wav':
                    with wave.open(str(target), 'wb') as writer:
                        writer.setnchannels(2)
                        writer.setsampwidth(2)
                        writer.setframerate(44100)
                        writer.writeframes(struct.pack('<2h', 1000, 3000) * 44100)
                else:
                    shutil.copyfile(fixtures / f'stream_probe.{extension}', target)
                metadata = {'metadata': {
                    'guid': {'type': 'string', 'value': guid},
                    'extension': {'type': 'string', 'value': '.' + extension},
                    'file_path': {'type': 'string', 'value': authored},
                    'resource_type': {'type': 'enum infernux::ResourceType', 'value': 'Audio'},
                    'force_mono': {'type': 'bool', 'value': mono},
                    'load_type': {'type': 'string', 'value':
                                  'streaming' if streaming else 'decompress_on_load'},
                }}
                artifact = {'runtime_artifact_id': artifact_id, 'runtime_path': relative,
                            'asset_guid': guid, 'package': 'Content.inxpkg', 'dependencies': []}
                records.append({'guid': guid, 'runtime_path': authored, 'metadata': metadata,
                                'primary_runtime_artifact_id': artifact_id,
                                'runtime_artifact_ids': [artifact_id],
                                'runtime_artifacts': [artifact]})
                catalog.append(artifact)
                cases.append((authored, target, mono, streaming))
    document = {'$schema': 'infernux.runtime_asset_records', 'entries': records}
    (player / 'Library/RuntimeAssetRecords.json').write_text(
        json.dumps(document, ensure_ascii=False), encoding='utf-8')
    assert not (player / 'Assets').exists()
    assert not list(player.rglob('*.meta'))
    PreferencesStore()._path = str(root / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    failures = []
    try:
        engine.init_headless(str(player))
        inx.Application._bind_engine(engine, 'player')
        runtime = PlayerRuntimeAssetCatalog.from_documents(
            str(player), {'artifacts': catalog}, document)
        project_context.set_runtime_asset_query(runtime.query_asset_guids)
        project_context.set_runtime_asset_resolver(runtime.resolve_guid)
        for authored, target, mono, streaming in cases:
            physical = inx.Application.asset_path(authored)
            assert Path(physical).resolve() == target.resolve()
            clip = inx.AudioClip.load(physical)
            assert isinstance(clip, inx.AudioClip), authored
            try:
                actual = (clip.channels, clip.is_streaming)
                expected = (1 if mono else 2, streaming)
                if actual != expected:
                    failures.append({'asset': authored, 'expected': expected, 'actual': actual})
                assert clip.is_loaded and clip.duration > 0, authored
            finally:
                clip.unload()
            assert not clip.is_loaded, authored
        (root / 'public-audio-observations.json').write_text(json.dumps({
            'cases': len(cases), 'failures': failures,
            'installed_package': inx.__file__, 'python': sys.executable,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        assert not failures, failures
    finally:
        engine.exit()


if __name__ == '__main__':
    exercise_public_player_audio(Path(sys.argv[1]))
