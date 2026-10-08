"""The Hub's default RenderStack is a portable, typed authoring document."""
import json
import ast
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_hub_template_uses_the_current_forward_parameter_document():
    from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline
    from infernux.renderstack.render_stack import RenderStack

    document = json.loads((ROOT / 'python/infernux/templates/project/default_scene.json').read_text())
    stack = next(component['data'] for obj in document['objects'] for component in obj['components']
                 if component['type_id'].endswith(':RenderStack'))
    assert stack['pipeline_class_name'] == RenderStack.DEFAULT_PIPELINE_NAME
    assert json.loads(stack['pipeline_params_json']) == {
        '__default__': RenderStack._encode_pipeline_params(DefaultForwardPipeline())}


def test_web_restores_and_edits_a_stack_without_desktop_callback_imports(tmp_path):
    script = '''
import importlib.abc
import json, sys
from pathlib import Path
import infernux.lib as native_api
from infernux.lib import _Infernux
_Infernux.__runtime_profile__ = 'web-player'
del native_api.RenderPipelineCallback
class NoDesktopPipeline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'infernux.renderstack.render_pipeline':
            raise AssertionError('Web parameter deserialization imported desktop callbacks')
sys.meta_path.insert(0, NoDesktopPipeline())
from infernux.renderstack import RenderStack
from infernux.renderstack.forward_parameters import MSAASamples
document = json.loads(Path(sys.argv[1]).read_text())
data = next(c['data'] for o in document['objects'] for c in o['components'] if c['type_id'].endswith(':RenderStack'))
data['effect_slots'] = []
stack = RenderStack()
stack._deserialize_fields_document(data)
assert stack.pipeline.msaa_samples is MSAASamples.X4
stack.set_pipeline_parameter('msaa_samples', MSAASamples.OFF)
assert stack.pipeline.msaa_samples is MSAASamples.OFF
assert json.loads(stack._serialize_fields_document()['pipeline_params_json'])['__default__']['msaa_samples']['name'] == 'OFF'
assert 'infernux.renderstack.render_pipeline' not in sys.modules
stack.set_pipeline('Unsupported')
try:
    stack.pipeline
except RuntimeError as error:
    assert 'requires the Default Forward' in str(error)
else:
    raise AssertionError('Unsupported Web pipeline silently accepted')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT/'python/infernux/templates/project/default_scene.json')],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT/'python')), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _web_settings_functions():
    source = ROOT/'external/plugins/infernux_web/native/bootstrap.py'
    names = {'infernux_web_render_settings', '_iter_web_render_effects'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    namespace = {'Any': Any, 'json': json, '_prepare_player_asset_contract': lambda: None}
    exec(compile(ast.Module([node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], []), str(source), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('name,samples', [('OFF', 1), ('X4', 4)])
def test_web_settings_decode_current_parameter_enum(name, samples):
    namespace = _web_settings_functions()
    namespace['_web_render_stack_document'] = lambda: {
        'pipeline_class_name': 'Default Forward', 'pipeline_params_json': json.dumps({
            '__default__': {'msaa_samples': {'$type': 'enum', 'enum_type': 'MSAASamples', 'name': name}}}),
        'effect_slots': []}
    assert namespace['infernux_web_render_settings']()['msaa_samples'] == samples


@pytest.mark.parametrize('value', [
    {'$type': 'enum', 'enum_type': 'MSAASamples', 'name': 'X2'},
    {'$type': 'enum', 'enum_type': 'OtherEnum', 'name': 'X4'},
    {'__enum_name__': 'X4'},
])
def test_web_settings_reject_unsupported_or_noncanonical_samples(value):
    namespace = _web_settings_functions()
    namespace['_web_render_stack_document'] = lambda: {
        'pipeline_class_name': 'Default Forward',
        'pipeline_params_json': json.dumps({'__default__': {'msaa_samples': value}})}
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        namespace['infernux_web_render_settings']()


def test_web_settings_read_cooked_effect_group_and_child_by_guid(tmp_path):
    from infernux.core.asset_document import encode_asset_document
    documents = {
        'a'*32: {'$schema': 'infernux.render_effect_group', 'entries': [
            {'asset': {'guid': 'b'*32}, 'enabled': True, 'overrides': {'intensity': .25}}]},
        'b'*32: {'$schema': 'infernux.render_effect', 'feature_type': 'infernux.post.bloom',
                 'parameters': {'intensity': .8, 'threshold': 1.5}},
    }
    for guid, document in documents.items():
        (tmp_path / (guid + '.inxdoc')).write_bytes(encode_asset_document(document))
    namespace = _web_settings_functions()
    namespace['_web_render_effect_path'] = lambda guid, hint: str(tmp_path/(guid+'.inxdoc'))
    assert list(namespace['_iter_web_render_effects']({'guid': 'a'*32})) == [
        ('infernux.post.bloom', {'intensity': .25, 'threshold': 1.5})]


@pytest.mark.parametrize('mode', ['scene_only', 'valid_script', 'bad_script', 'no_scene'])
def test_web_content_accepts_scriptless_scene_but_keeps_script_abi_validation(tmp_path, monkeypatch, mode):
    import importlib.util
    import marshal
    from types import CodeType, ModuleType
    from infernux.engine.player_package_native import extract_pack, read_entry, write_pack

    host = ModuleType('_InfernuxWebHost')
    host.extract_package = extract_pack
    host.read_entry = read_entry
    monkeypatch.setitem(sys.modules, host.__name__, host)
    player = tmp_path/'Player'
    data = player/'Game_Data'
    data.mkdir(parents=True)
    files, artifacts = [], []
    payloads = {'Library/Main.scene': b'{"objects": []}'} if mode != 'no_scene' else {}
    if mode != 'scene_only':
        payloads['Scripts/Main.pyc'] = (b'invalid' if mode == 'bad_script' else
            importlib.util.MAGIC_NUMBER + b'\0'*12 + marshal.dumps(compile('VALUE=1', 'Main.py', 'exec')))
    for index, (name, payload) in enumerate(payloads.items()):
        source = tmp_path/str(index)
        source.write_bytes(payload)
        files.append((name, source))
        artifacts.append({'logical_type': 'compiled_script' if name.endswith('.pyc') else 'scene_artifact',
                          'package': 'Game_Data/Content.inxpkg', 'runtime_path': name})
    package = data/'Content.inxpkg'
    write_pack(files, package)
    catalog = {'$schema': 'infernux.runtime_asset_catalog', 'player_host': {}, 'artifacts': artifacts,
        'packages': [{'path': 'Game_Data/Content.inxpkg', 'archive_bytes': package.stat().st_size,
                      'file_count': len(files), 'raw_bytes': sum(map(len, payloads.values())),
                      'stored_bytes': 0, 'codec': 'none'}]}
    catalog_source = tmp_path/'RuntimeAssetCatalog.json'
    catalog_source.write_text(json.dumps(catalog))
    write_pack([('RuntimeAssetCatalog.json', catalog_source)], data/'AssetCatalog.inxcat')
    path = ROOT/'external/plugins/infernux_web/native/bootstrap.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_RUNTIME_ASSET_CATALOG_SCHEMA', '_RUNTIME_ASSET_CATALOG_FIELDS', '_RUNTIME_PACKAGE_FIELDS'}
    nodes = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in {'_validate_runtime_asset_catalog', '_prepare_cooked_player_content'})]
    namespace = dict(Any=Any, os=os, json=json, marshal=marshal, importlib=importlib, CodeType=CodeType, _player_root=str(player))
    exec(compile(ast.Module(nodes, []), str(path), 'exec'), namespace)
    if mode in {'bad_script', 'no_scene'}:
        with pytest.raises(RuntimeError, match='script ABI mismatch' if mode == 'bad_script' else 'at least one scene'):
            namespace['_prepare_cooked_player_content']()
    else:
        assert namespace['_prepare_cooked_player_content']() == str(data)
        assert (data/'Library/Main.scene').read_bytes() == payloads['Library/Main.scene']
