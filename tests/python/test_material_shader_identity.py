"""Material annotations follow imported stage GUIDs through package and reload changes."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('location', ['assets', 'package'])
@pytest.mark.parametrize('case', ['initial', 'refresh', 'same_name', 'missing', 'restored', 'rename',
                                  'empty', 'switch_guid', 'stable_values', 'disk'])
def test_material_annotation_stage_identity(tmp_path, location, case):
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), str(tmp_path), location, case],
        env=dict(os.environ), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'MATERIAL_SHADER_IDENTITY_OK' in result.stdout


def exercise(project, location, case):
    import copy
    import json
    from types import SimpleNamespace
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.ui import inspector_shader_utils as shaders
    from infernux.engine.ui.inspector_material import _sync_shader_annotations
    from infernux.engine.ui.asset_details_renderer import _load_material, _sync_material_shader_metadata
    from infernux.lib import InxMaterial, LogLevel, RuntimeMode

    assets = project / 'Assets'
    assets.mkdir()
    (project / 'ProjectSettings').mkdir()
    package = project / 'Packages' / 'test' / 'shader'
    (package / 'runtime').mkdir(parents=True)
    (package / 'inx_package.json').write_text(json.dumps({
        'reference': 'test/shader', 'name': 'TestShader', 'version': '1.0.0',
    }), encoding='utf-8')
    vertex = assets / 'Vertex.vert'
    fragment = (assets if location == 'assets' else package / 'runtime') / 'Fragment.frag'
    vertex.write_text('#version 450\nShaderInfo {\n Name "TestVertex"\n Properties {\n Float keep = 1.0\n }\n}\n', encoding='utf-8')
    source = ('#version 450\nShaderInfo {\n Name "TestFragment"\n ShadingModel Unlit\n'
              ' Properties {\n Float packageGain = 0.75\n }\n}\n'
              'void surface(out SurfaceData s) { s = InitSurfaceData(); s.albedo = vec3(material.packageGain); }\n')
    fragment.write_text(source, encoding='utf-8')
    if case in ('same_name', 'switch_guid'):
        decoy = assets / 'AAA_Decoy.frag'
        decoy.write_text(source.replace('packageGain', 'wrongGain'), encoding='utf-8')
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        vertex_guid = database.get_guid_from_path(str(vertex))
        fragment_guid = database.get_guid_from_path(str(fragment))
        assert vertex_guid and fragment_guid
        shaders.bump_shader_property_generation()
        native = InxMaterial.create_default_unlit()
        document = native.serialize_document()
        document['shaders'] = {'vertex': {'guid': vertex_guid, 'shader_id': 'TestVertex'},
                               'fragment': {'guid': fragment_guid, 'shader_id': 'TestFragment'}}
        document['properties'] = {'keep': {'type': 0, 'value': 1.0}, 'packageGain': {'type': 0, 'value': .375}}
        assert native.deserialize_document(document)
        document = native.serialize_document()
        state = SimpleNamespace(extra={})
        _sync_shader_annotations(document, state)
        assert any(value == 'TestFragment' for _, value in shaders.get_shader_candidates('.frag'))
        if case in ('refresh', 'missing', 'restored', 'rename'):
            before = copy.deepcopy(document)
            if case in ('missing', 'restored'):
                sidecar = Path(str(fragment) + '.meta')
                metadata = sidecar.read_bytes()
                fragment.unlink()
                assert database.delete_asset(str(fragment)).succeeded
                shaders.bump_shader_property_generation()
                _sync_shader_annotations(document, state)
                assert document == before, 'an unavailable stage must not publish a partial schema'
                if case == 'restored':
                    fragment.write_text(source, encoding='utf-8')
                    sidecar.write_bytes(metadata)
                    assert database.import_asset(str(fragment)).guid == fragment_guid
            elif case == 'rename':
                moved = fragment.with_name('Renamed.frag')
                fragment.replace(moved)
                assert database.move_asset(str(fragment), str(moved)).succeeded
                moved.write_text(source.replace('TestFragment', 'RenamedFragment'), encoding='utf-8')
                assert database.reimport_asset(str(moved)).succeeded
            shaders.bump_shader_property_generation()
            _sync_shader_annotations(document, state)
        if case == 'empty':
            vertex.write_text('#version 450\nShaderInfo { Name "TestVertex" }\n', encoding='utf-8')
            fragment.write_text('#version 450\nShaderInfo { Name "TestFragment" ShadingModel Unlit }\n'
                                'void surface(out SurfaceData s) { s = InitSurfaceData(); }\n', encoding='utf-8')
            assert database.reimport_asset(str(vertex)).succeeded
            assert database.reimport_asset(str(fragment)).succeeded
            shaders.bump_shader_property_generation()
            _sync_shader_annotations(document, state)
            assert document['properties'] == {}
            assert shaders.get_material_property_display_order(document) == []
            # The first call validates the newly published property names;
            # subsequent stable frames must recognize an empty schema too.
            _sync_shader_annotations(document, state)
            get_names = shaders.get_all_shader_property_names
            calls = []
            def counted_names(*args):
                calls.append(args)
                return get_names(*args)
            shaders.get_all_shader_property_names = counted_names
            try:
                for _ in range(50):
                    _sync_shader_annotations(document, state)
                assert calls == []
            finally:
                shaders.get_all_shader_property_names = get_names
            return
        if case == 'switch_guid':
            document['shaders']['fragment']['guid'] = database.get_guid_from_path(str(decoy))
            _sync_shader_annotations(document, state)
            assert shaders.get_material_property_display_order(document) == ['keep', 'wrongGain']
            assert document['properties']['wrongGain']['value'] == pytest.approx(.75)
            return
        if case == 'stable_values':
            parse = shaders.parse_shader_properties
            calls = []
            def counted_parse(path):
                calls.append(path)
                return parse(path)
            shaders.parse_shader_properties = counted_parse
            try:
                for _ in range(50):
                    document['properties']['packageGain']['value'] += 1
                    _sync_shader_annotations(document, state)
                assert document['properties']['packageGain']['value'] == pytest.approx(50.375)
                assert calls == [], 'value edits must reuse the published schema'
            finally:
                shaders.parse_shader_properties = parse
            document['properties']['packageGain']['value'] = .375
        assert document['properties']['packageGain']['value'] == pytest.approx(.375)
        assert 'packageGain' in shaders.get_material_property_display_order(document)
        assert 'wrongGain' not in document['properties']
        loaded = copy.deepcopy(document)
        _sync_material_shader_metadata(loaded)
        assert loaded['properties']['packageGain']['value'] == pytest.approx(.375)
        assert 'wrongGain' not in loaded['properties']
        if case != 'missing':
            assert native.deserialize_document(loaded)
            reopened = InxMaterial.create_default_unlit()
            assert reopened.deserialize_document(native.serialize_document())
            assert reopened.serialize_document()['properties']['packageGain']['value'] == pytest.approx(.375)
        if case == 'disk':
            material_path = assets / 'Authored.mat'
            native.save_to(str(material_path))
            assert database.import_asset(str(material_path)).succeeded
            result = _load_material(str(material_path))
            assert result is not None
            material, extra = result
            assert extra['cached_data']['properties']['packageGain']['value'] == pytest.approx(.375)
            assert 'packageGain' in shaders.get_material_property_display_order(extra['cached_data'])
            assert material.save(str(material_path))
            saved = json.loads(material_path.read_text(encoding='utf-8'))
            assert saved['properties']['packageGain']['value'] == pytest.approx(.375)
            assert saved['shaders']['fragment']['guid'] == fragment_guid
            from_disk = InxMaterial.create_default_unlit()
            assert from_disk.deserialize_document(saved)
            assert from_disk.serialize_document()['properties']['packageGain']['value'] == pytest.approx(.375)
    finally:
        shaders.bump_shader_property_generation()
        engine.exit()


if __name__ == '__main__':
    exercise(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
    print('MATERIAL_SHADER_IDENTITY_OK')
