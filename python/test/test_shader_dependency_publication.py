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
        paths.append(path)
        return path,imported.guid
    yield prefix,create,registry
    for path in reversed(paths):
        result=AssetManager.delete_asset(str(path))
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
