"""Inspector projections use the identity of genuine imported project assets."""
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from infernux.core.assets import AssetManager
from infernux.core.asset_types import (
    SpriteFrame, TextureImportSettings, TextureType, TextureCompression,
    read_meta_file, read_texture_import_settings, write_texture_import_settings,
)
from infernux.engine.interaction import EditorInteractionCore, SelectionDomain, SelectionTarget
from infernux.engine.ui import asset_details_renderer as ui
from infernux.engine.ui.core_panel_interactions import inspector_panel_interaction
from infernux.renderstack.render_effect_asset import (
    RenderEffectAsset, RenderEffectGroupAsset, RenderEffectGroupEntry,
    EffectAssetReference, dump_render_effect_document,
)
from infernux.renderstack.render_effect_compiler import RenderEffectArtifactRegistry


@pytest.fixture
def project_assets(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    core = EditorInteractionCore()
    core.project_assets.configure(database.project_root, database)
    core.panels.register_type('inspector', inspector_panel_interaction(lambda: None))
    core.panels.bind_view('inspector', 'inspector', object())
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    try:
        yield SimpleNamespace(database=database, core=core, folder=folder)
    finally:
        core.shutdown()
        RenderEffectArtifactRegistry.clear()


@pytest.mark.parametrize('action', [
    'first', 'second', 'reorder', 'remove', 'rename', 'asset_only', 'other_asset', 'missing_meta', 'guid_case',
])
def test_sprite_frame_selection_roundtrip(project_assets, action):
    project = project_assets
    path = project.folder / 'Sprite.png'
    Image.new('RGBA', (8, 4), (255, 0, 0, 255)).save(path)
    result = project.database.import_asset(str(path))
    assert result.succeeded, result.error
    frames = [SpriteFrame(name='left', x=0, y=0, w=4, h=4), SpriteFrame(name='right', x=4, y=0, w=4, h=4)]
    settings = TextureImportSettings(texture_type=TextureType.SPRITE, compression=TextureCompression.NONE,
                                    generate_mipmaps=False, sprite_frames=frames)
    assert write_texture_import_settings(str(path), settings)
    assert project.database.reimport_asset(str(path)).succeeded
    state = ui._State()
    state.file_path, state.category = str(path), 'texture'
    state.meta = read_meta_file(str(path))
    state.settings = settings = read_texture_import_settings(str(path))
    assert [frame.stable_id for frame in settings.sprite_frames] == [frame.stable_id for frame in frames]
    initial = 0 if action == 'first' else 1
    ui._select_sprite_frame(state, settings.sprite_frames[initial])
    selected = project.core.selection.snapshot.primary
    assert selected.document_id == result.guid and selected.target_id == frames[initial].stable_id
    expected = initial
    if action == 'reorder':
        settings.sprite_frames.reverse()
        expected = 0
    elif action == 'remove':
        settings.sprite_frames.pop(initial)
        expected = -1
    elif action == 'rename':
        moved = project.folder / 'Renamed Sprite.png'
        path.replace(moved)
        assert project.database.move_asset(str(path), str(moved))
        state.file_path, state.meta = str(moved), read_meta_file(str(moved))
        assert state.meta['guid'] == result.guid
    elif action == 'asset_only':
        ui._select_sprite_frame(state, None)
        expected = -1
    elif action == 'other_asset':
        other = project.folder / 'Other.png'
        Image.new('RGBA', (8, 4)).save(other)
        imported = project.database.import_asset(str(other))
        assert imported.succeeded
        project.core.selection.select(SelectionTarget.asset_subresource(
            imported.guid, frames[1].stable_id, sub_kind='sprite_frame'), owner_id='inspector')
        expected = -1
    elif action == 'missing_meta':
        state.meta = None
        expected = -1
    elif action == 'guid_case':
        state.meta['guid'] = result.guid.upper()
    before = project.core.selection.snapshot
    assert ui._selected_sprite_frame_index(state, settings) == expected
    after = project.core.selection.snapshot
    if action == 'remove':
        assert after.primary.domain is SelectionDomain.ASSET
        assert after.primary == SelectionTarget.asset(result.guid)
    else:
        assert after == before


class EntryLookupCompleted(Exception):
    pass


class SectionSink:
    """Stop explicitly after production lookup, before entry widgets."""
    def __init__(self, expanded):
        self.expanded, self.headers = expanded, 0

    def get_dpi_scale(self):
        return 1.0

    def render_compact_section_header(self, *args):
        self.headers += 1
        if self.headers == 2:
            raise EntryLookupCompleted
        return self.expanded


@pytest.mark.parametrize('expanded', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_real_effect_group_lookup_from_inspector(project_assets, expanded, nested):
    project = project_assets
    path = project.folder / ('Referenced.effectgroup' if nested else 'Bloom.effect')
    asset = RenderEffectGroupAsset() if nested else RenderEffectAsset('infernux.post.bloom')
    path.write_text(dump_render_effect_document(asset), encoding='utf-8')
    result = project.database.import_asset(str(path))
    assert result.succeeded, result.error
    group_path = project.folder / 'Post.effectgroup'
    group_path.write_text(dump_render_effect_document(RenderEffectGroupAsset(entries=(
        RenderEffectGroupEntry('referenced', EffectAssetReference(result.guid)),
    ))), encoding='utf-8')
    assert project.database.import_asset(str(group_path)).succeeded
    state = ui._State()
    state.file_path = str(group_path)
    state.settings, state.extra = ui._load_render_effect(str(group_path))
    before = state.settings.serialize_document()
    disk = group_path.read_bytes()
    ctx = SectionSink(expanded)
    if expanded:
        with pytest.raises(EntryLookupCompleted):
            ui._render_render_effect_body(ctx, None, state)
    else:
        ui._render_render_effect_body(ctx, None, state)
    assert ctx.headers == (2 if expanded else 1)
    assert Path(AssetManager._get_path_from_guid(result.guid)) == path
    assert state.settings.serialize_document() == before and group_path.read_bytes() == disk
