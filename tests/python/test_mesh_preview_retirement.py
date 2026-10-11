import io

import pytest
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial
from tests.gpu.mesh_preview_retirement_case import MeshPreviewRetirementCase
from test_material_preview_retirement import pump_until


@pytest.mark.parametrize('stage', ['queued','loading','ready'])
@pytest.mark.parametrize('recreate', [False,True])
def test_deleted_mesh_retires_preview_work(engine, scene, tmp_path, monkeypatch, stage, recreate):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    console = ConsolePanel()
    case = None
    try:
        case = MeshPreviewRetirementCase(engine,tmp_path/'Model',stage,recreate)
        pump_until(engine,lambda _:case.tick()['done'])
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        if case is not None:
            case.close()


@pytest.mark.parametrize('published', [False,True])
def test_source_retirement_covers_embedded_products_but_not_a_sibling_prefix(engine, scene, tmp_path, published):
    # Real native task keys: retiring a model owns embedded products too,
    # while a distinct filename that starts with the same bytes remains live.
    source = tmp_path/'EmbeddedModel.fbx'
    textures = ['tex|'+str(source), 'tex|'+str(source)+'::subtex:0', 'document|'+str(source)]
    material = 'mat|'+str(source)+'::submat:0'
    control = 'tex|'+str(source)+'_Other'
    output = io.BytesIO()
    Image.new('RGB',(8,8),'red').save(output,format='JPEG')
    for key in textures+[control]:
        assert engine.schedule_texture_preview_from_memory(key,output.getvalue(),1,False)
    engine.query_or_schedule_material_preview(material,'',InxMaterial.create_default_lit().serialize(),0,True)
    try:
        if published:
            pump_until(engine,lambda _:all(engine.get_texture_preview_texture_id(k) for k in textures+[control])
                       and engine.get_material_preview_texture_id(material))
        engine.release_asset_preview_tasks(str(source))
        assert all(engine.get_texture_preview_texture_id(k) == 0 for k in textures)
        assert engine.get_material_preview_texture_id(material) == 0
        pump_until(engine,lambda frame:frame >= 20 and engine.get_texture_preview_texture_id(control))
        assert all(engine.get_texture_preview_texture_id(k) == 0 for k in textures)
        assert engine.get_material_preview_texture_id(material) == 0
    finally:
        engine.release_asset_preview_tasks(str(source))
        engine.release_texture_preview_task(control)
