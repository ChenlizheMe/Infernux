"""Dependency edits publish one complete set of real native shader candidates."""
import json
from pathlib import Path
import uuid

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.core.shader import Shader
from infernux.lib import AssetMutationErrorCode, AssetRegistry, InxMaterial


@pytest.fixture
def dependency_sources(engine, monkeypatch):
    database=engine.get_asset_database()
    registry=AssetRegistry.instance()
    monkeypatch.setattr(AssetManager,"_engine",engine)
    monkeypatch.setattr(AssetManager,"_asset_database",database)
    monkeypatch.setattr(AssetManager,"_registry",registry)
    root=Path(database.assets_root)
    root.mkdir(parents=True,exist_ok=True)
    prefix="Dependency_"+uuid.uuid4().hex
    paths=[]
    def create(label,extension,source):
        path=root/(prefix+label+extension)
        path.write_text(source,encoding="utf-8")
        imported=AssetManager.import_asset(str(path))
        assert imported,imported.error
        paths.append((path,imported.guid))
        return path,imported.guid
    yield prefix,create,registry
    for _,guid in reversed(paths):
        path=database.get_path_from_guid(guid)
        if path:
            result=AssetManager.delete_asset(path)
            assert result,result.error


def sources(prefix):
    name=prefix+" Library"
    good=(f'ShaderInfo {{ Name "{name}" }}\n'
          'vec4 sharedColor(){return vec4(0.2,0.3,0.4,1);}\n'
          'vec4 optionalColor(){return sharedColor();}\n')
    bad=good.replace('0.2,0.3,0.4','0.7,0.6,0.5').replace(
        'vec4 optionalColor(){return sharedColor();}\n','')
    restored=good.replace('0.2,0.3,0.4','0.7,0.6,0.5')
    return name,good,bad,restored


def test_standalone_dependency_failure_preserves_every_published_generation(engine,dependency_sources):
    prefix,create,registry=dependency_sources
    library_name,good,bad,restored=sources(prefix)
    library,_=create('Library','.glsl',good)
    stages=[]
    for label,function in [('A','sharedColor'),('B','optionalColor')]:
        name=prefix+label
        path,guid=create(label,'.frag',
            f'#version 450\nShaderInfo {{ Name "{name}" Capabilities [Fullscreen] '
            f'Imports ["{library_name}"] Outputs {{ Float4 outColor }} }}\n'
            f'void main(){{outColor={function}();}}\n')
        assert Shader.reload(str(path))
        stages.append((guid,name))
    versions=[registry.get_asset_version(guid) for guid,_ in stages]
    assert all(versions)
    library.write_text(bad,encoding='utf-8')
    rejected=AssetManager.reimport_asset(str(library))
    assert not rejected and rejected.database_committed
    assert rejected.error_code==AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    assert 'optionalColor' in rejected.error
    # The valid A candidate must wait for the rejected B candidate.
    assert [registry.get_asset_version(guid) for guid,_ in stages]==versions
    assert all(engine.is_shader_loaded(name,'fragment') for _,name in stages)
    library.write_text(restored,encoding='utf-8')
    accepted=AssetManager.reimport_asset(str(library))
    assert accepted,accepted.error
    assert [registry.get_asset_version(guid) for guid,_ in stages]==[v+1 for v in versions]


def test_linked_dependency_failure_does_not_retire_a_successful_sibling_program(engine,dependency_sources):
    prefix,create,registry=dependency_sources
    library_name,good,bad,restored=sources(prefix)
    library,_=create('Library','.glsl',good)
    vertex_name=prefix+' Vertex'
    _,vertex_guid=create('Vertex','.vert',f'#version 450\nShaderInfo {{ Name "{vertex_name}" }}\n')
    materials=[]
    for label,function in [('A','sharedColor'),('B','optionalColor')]:
        name=prefix+label
        _,fragment_guid=create(label,'.frag',
            f'#version 450\nShaderInfo {{ Name "{name}" ShadingModel Unlit Imports ["{library_name}"] }}\n'
            f'void surface(out SurfaceData s){{s=InitSurfaceData();s.albedo={function}().rgb;}}\n')
        document=InxMaterial.create_default_lit().serialize_document()
        document['builtin']=False
        document['name']=name
        document['shaders']={'vertex':{'guid':vertex_guid,'shader_id':vertex_name},
                             'fragment':{'guid':fragment_guid,'shader_id':name}}
        _,guid=create(label,'.mat',json.dumps(document))
        material=AssetManager.load(guid,Material)
        assert engine.refresh_material_pipeline(material._native)
        materials.append(material)
    retired=engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']
    library.write_text(bad,encoding='utf-8')
    rejected=AssetManager.reimport_asset(str(library))
    assert not rejected and 'optionalColor' in rejected.error
    assert engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']==retired
    for material in materials:
        assert engine.refresh_material_pipeline(material._native)

    assert engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']==retired
    library.write_text(restored,encoding='utf-8')
    accepted=AssetManager.reimport_asset(str(library))
    assert accepted,accepted.error
    assert engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']>retired


@pytest.mark.parametrize('header',('same-name','renamed-name'))
def test_dependency_move_keeps_guid_and_reports_rejected_runtime_candidate(engine,dependency_sources,header):
    prefix,create,registry=dependency_sources
    library_name,good,bad,_=sources(prefix)
    library,library_guid=create('Library','.glsl',good)
    name=prefix+' Stage'
    stage,stage_guid=create('Stage','.frag',
        f'#version 450\nShaderInfo {{ Name "{name}" Capabilities [Fullscreen] '
        f'Imports ["{library_name}"] Outputs {{ Float4 outColor }} }}\n'
        'void main(){outColor=optionalColor();}\n')
    assert Shader.reload(str(stage))
    version=registry.get_asset_version(stage_guid)
    if header=='renamed-name':
        bad=good.replace(library_name,library_name+' Renamed')
    library.write_text(bad,encoding='utf-8')
    moved_path=library.with_name(library.stem+'Moved.glsl')
    library.replace(moved_path)
    result=AssetManager.move_asset(str(library),str(moved_path))
    assert result.database_committed and not result
    assert result.error_code==AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    assert registry.get_asset_version(stage_guid)==version
    database=engine.get_asset_database()
    assert database.get_guid_from_path(str(moved_path))==library_guid
    assert not database.get_guid_from_path(str(library))
    assert ('optionalColor' if header=='same-name' else 'shader import not found') in result.error
    moved_path.write_text(good,encoding='utf-8')
    restored=AssetManager.reimport_asset(str(moved_path))
    assert restored,restored.error
    assert registry.get_asset_version(stage_guid)==version+1
    moved_path.replace(library)
    back=AssetManager.move_asset(str(moved_path),str(library))
    assert back,back.error
    assert database.get_guid_from_path(str(library))==library_guid


def test_same_name_vertex_save_rejects_all_pairs_before_retiring_any_program(engine,dependency_sources):
    prefix,create,_=dependency_sources
    name=prefix+' Vertex'
    original=(f'#version 450\nShaderInfo {{ Name "{name}" Outputs {{ Smooth Float signal }} }}\n'
              'VertexOutput vertex(inout VertexInput v){VertexOutput o;o.signal=0.25;return o;}\n')
    changed=(f'#version 450\nShaderInfo {{ Name "{name}" }}\n'
             'void vertex(inout VertexInput v){v.position.x+=0.2;}\n')
    vertex,vertex_guid=create('Vertex','.vert',original)
    materials=[]
    for label in ('A','B'):
        fragment_name=prefix+label
        inputs=' Inputs { Smooth Float signal }' if label=='B' else ''
        color='vec3(fragmentInput.signal)' if label=='B' else 'vec3(0.2,0.3,0.4)'
        _,fragment_guid=create(label,'.frag',
            f'#version 450\nShaderInfo {{ Name "{fragment_name}" ShadingModel Unlit{inputs} }}\n'
            f'void surface(out SurfaceData s){{s=InitSurfaceData();s.albedo={color};}}\n')
        document=InxMaterial.create_default_lit().serialize_document()
        document['builtin']=False
        document['name']=fragment_name
        document['shaders']={'vertex':{'guid':vertex_guid,'shader_id':name},
                             'fragment':{'guid':fragment_guid,'shader_id':fragment_name}}
        _,guid=create(label,'.mat',json.dumps(document))
        material=AssetManager.load(guid,Material)
        assert engine.refresh_material_pipeline(material._native)
        materials.append(material)
    before=[material._native.serialize_document() for material in materials]
    retired=engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']
    vertex.write_text(changed,encoding='utf-8')
    rejected=AssetManager.reimport_asset(str(vertex))
    assert not rejected and rejected.database_committed and 'signal' in rejected.error
    assert engine.gpu_residency_snapshot['shader_hot_reload_retirement_count']==retired
    assert [material._native.serialize_document() for material in materials]==before
    vertex.write_text(original,encoding='utf-8')
    restored=AssetManager.reimport_asset(str(vertex))
    assert restored,restored.error
    for material in materials:
        assert engine.refresh_material_pipeline(material._native)
