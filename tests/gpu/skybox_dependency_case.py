"""A cold authored skybox must become resident before Scene publication."""
from pathlib import Path

import numpy as np

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction
from infernux.lib import AssetRegistry


def create_skybox_asset(directory, database):
    source = Path(directory) / 'Skybox.mat'
    source.parent.mkdir(parents=True, exist_ok=True)
    builtin = AssetRegistry.instance().get_builtin_material('SkyboxProcedural')
    assert builtin is not None
    material = Material(builtin.clone())
    for key in ('skyTopColor','skyHorizonColor','groundColor'):
        material.set_color(key,.8,.02,.01)
    material.set_float('exposure',1.)
    assert material.save(str(source))
    imported = AssetManager.import_asset(str(source),database=database)
    assert imported.succeeded, imported.error
    return source, imported.guid


def publish_skybox(scene, guid, database, engine, snapshot):
    registry = AssetRegistry.instance()
    assert not registry.is_loaded(guid)
    scene.set_environment({'skybox_material_guid':guid})
    document = scene._capture_play_mode_snapshot() if snapshot else scene.serialize_document()
    scene.set_environment({'skybox_material_guid':''})
    observations = []

    def before_commit():
        # The old live environment cannot have lazily loaded this material.
        assert scene.get_environment()['skybox_material_guid'] == ''
        assert registry.is_loaded(guid), 'Skybox must be resident before native publication'
        observations.append(guid)

    transaction = SceneDocumentTransaction(scene,document=document,asset_database=database,
        native_engine=engine,clear_registries=False,before_commit=before_commit)
    assert transaction.run_to_completion(), transaction.error
    assert observations == [guid]
    assert scene.get_environment()['skybox_material_guid'] == guid
    return dict(before_commit_resident=observations, snapshot=snapshot, status=transaction.status)


def observe_skybox(pixels):
    rgb = pixels[...,:3].astype(np.float32)
    assert np.isfinite(rgb).all()
    red = (rgb[...,0] > rgb[...,1] + .2) & (rgb[...,0] > rgb[...,2] + .2)
    assert red.all(), f'Expected authored red skybox, observed only {red.mean():.1%} red pixels'
    return dict(red_pixels=int(red.sum()),shape=list(pixels.shape),mean=rgb.mean(axis=(0,1)).tolist())
