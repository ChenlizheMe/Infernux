"""Native Inspector headers keep live textures separate from immutable metadata."""
from collections import Counter
from pathlib import Path
import time

import pytest

from infernux.lib import (
    InspectorComponentInfo, InspectorObjectInfo, InspectorPanel,
    InspectorRevisionSnapshot, InspectorTransformData, RenderPipelineCallback,
    get_gui_semantic_snapshot, set_gui_semantic_capture_enabled,
)


@pytest.mark.parametrize('multi', [False, True])
def test_component_icons_stay_live_across_cached_metadata_packets(engine, scene, multi):
    objects = [scene.create_game_object('Icon publication A')]
    if multi:
        objects.append(scene.create_game_object('Icon publication B'))
    cameras = {obj.id: obj.add_component('Camera') for obj in objects}
    by_id = {obj.id: obj for obj in objects}
    panel = InspectorPanel()
    revision = InspectorRevisionSnapshot()
    revision.target = revision.schema = revision.value = revision.preview = 1
    panel.get_revision_snapshot = lambda: revision
    panel.is_multi_selection = lambda: multi
    panel.get_selected_ids = lambda: [obj.id for obj in objects]
    panel.set_selected_object_id(objects[0].id)
    metadata_reads = Counter()
    icon_reads = Counter()
    snapshots = []
    first_ready = {}

    def object_info(object_id):
        obj = by_id[object_id]
        info = InspectorObjectInfo()
        info.name = obj.name
        info.active = True
        info.tag = 'Untagged'
        info.transform_component_id = obj.get_transform().component_id
        return info

    def component_list(object_id):
        metadata_reads[object_id] += 1
        camera = cameras[object_id]
        info = InspectorComponentInfo()
        info.type_name = camera.type_name
        info.component_id = camera.component_id
        info.enabled = camera.enabled
        info.is_native = True
        # The first packet is published before upload; a later schema packet
        # captures a valid handle. Neither cached packet owns texture lifetime.
        info.icon_id = (engine.get_texture_preview_texture_id('test.inspector.live_icon.Camera')
                        if revision.schema == 2 else 0)
        return [info]

    def current_icon(type_name, _is_script):
        icon_reads[type_name] += 1
        source = Path(__file__).resolve().parents[2] / 'python/infernux/resources/icons/components'
        source /= 'component_' + type_name.lower() + '.png'
        texture_id, _width, _height = engine.query_or_schedule_texture_preview(
            'test.inspector.live_icon.' + type_name, str(source), source.stat().st_mtime_ns,
            nearest=True, srgb=False,
        )
        # Hold the initial two native GUI draws pending even on a fast device.
        if icon_reads[type_name] <= 2:
            return 0
        if texture_id:
            first_ready.setdefault(type_name, icon_reads[type_name])
        return texture_id

    panel.get_object_info = object_info
    panel.get_component_list = component_list
    panel.get_component_icon_id = current_icon
    panel.get_transform_data = lambda _object_id: InspectorTransformData()
    panel.get_all_tags = lambda: ['Untagged']
    panel.get_all_layers = lambda: ['Default']

    class GuiOnly(RenderPipelineCallback):
        def render(self, _context, _camera):
            pass

    deadline = time.monotonic() + 15
    previous_frame = int(get_gui_semantic_snapshot().get('frame', 0) or 0)

    def observe(_delta):
        nonlocal previous_frame
        snapshot = get_gui_semantic_snapshot()
        frame = snapshot.get('frame', 0)
        if frame and frame != previous_frame:
            previous_frame = frame
            targets = [item for item in snapshot.get('targets', [])
                       if item.get('kind') in {'component_icon', 'component_enabled', 'component_label'}]
            snapshots.append(targets)
            if len(snapshots) >= 4 and len(first_ready) == 2:
                revision.schema = 2
        if len(snapshots) >= 12 or time.monotonic() >= deadline:
            engine.exit()

    engine.set_render_pipeline(GuiOnly())
    engine.register_gui_renderable('test.inspector.live_icons', panel)
    engine.set_maximized(True)
    engine.show()
    set_gui_semantic_capture_enabled(True)
    try:
        engine.set_pre_scene_update_callback(observe)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        set_gui_semantic_capture_enabled(False)
        engine.unregister_gui_renderable('test.inspector.live_icons')
        engine.set_render_pipeline(None)
        engine.set_maximized(False)
        engine.hide()
        for name in ('Camera', 'Transform'):
            engine.release_texture_preview_task('test.inspector.live_icon.' + name)

    assert len(snapshots) >= 12, 'Twelve real native GUI frames did not complete'
    assert metadata_reads == {obj.id: revision.schema for obj in objects}
    for name in ('Camera', 'Transform'):
        assert icon_reads[name] >= 12, (name, icon_reads)
        assert name in first_ready, (name, first_ready)
        icons = [item for frame in snapshots for item in frame
                 if item['kind'] == 'component_icon' and item['label'] == name]
        assert any(item.get('value') is False for item in icons), name
        assert any(item.get('value') is True for item in icons), name
        assert len({tuple(item['rect'][2:]) for item in icons}) == 1, 'Upload changed icon size'
    separations = set()
    for frame in snapshots:
        camera = {item['kind']: item for item in frame if item['label'] == 'Camera'}
        assert {'component_icon', 'component_enabled', 'component_label'} <= camera.keys(), camera
        icon, checkbox, label = (camera[key]['rect'] for key in
                                 ('component_icon', 'component_enabled', 'component_label'))
        assert icon[0] + icon[2] < checkbox[0], camera
        assert checkbox[0] + checkbox[2] < label[0], camera
        separations.add((checkbox[0] - icon[0], label[0] - checkbox[0]))
    assert len(separations) == 1, 'Texture publication moved the checkbox or label'
