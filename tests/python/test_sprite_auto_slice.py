"""Invalid grids must not become authoring edits or queued texture imports."""
import pytest

from infernux.core.asset_types import SpriteFrame, TextureImportSettings, TextureType
from infernux.engine.interaction import AuthoringMutationService, DocumentRegistry, SelectionService, SelectionTarget
from infernux.engine.undo import UndoManager
from infernux.engine.ui import asset_details_renderer as details
from infernux.engine.ui.asset_import_progress import AssetImportProgressService


@pytest.mark.parametrize('width,height,rows,cols', [
    (4, 4, 1, 5), (4, 4, 5, 1), (4, 4, 5, 5), (4, 4, 0, 2),
    (4, 4, 2, 0), (0, 4, 1, 1), (4, 0, 1, 1), (4, 4, -1, 1),
])
def test_invalid_grid_preserves_document_selection_history_and_imports(monkeypatch, tmp_path, width, height, rows, cols):
    for cls in (DocumentRegistry, UndoManager, AuthoringMutationService, SelectionService, AssetImportProgressService):
        monkeypatch.setattr(cls, '_instance', None)
    registry, history, selection = DocumentRegistry(), UndoManager(), SelectionService()
    path = tmp_path / 'grid.png'
    path.write_bytes(b'source remains unchanged')
    meta = path.with_suffix('.png.meta')
    meta.write_bytes(b'authoring metadata remains unchanged')
    details._ensure_categories()
    state = details._State()
    state.file_path, state.category, state.meta = str(path), 'texture', {'guid': 'a' * 32}
    frame = SpriteFrame(name='Full', w=4, h=4)
    state.settings = TextureImportSettings(texture_type=TextureType.SPRITE, sprite_frames=[frame])
    state.disk_settings = state.settings.copy()
    details._bind_import_settings_document(state, details._categories['texture'])
    selection.select(SelectionTarget.asset_subresource('a' * 32, frame.stable_id, sub_kind='sprite_frame'),
                     owner_id='inspector', record_history=False)
    before = state.settings.to_dict(), selection.snapshot, registry.require(state.document_id).revision
    sprite = details._SpriteEditorState()
    sprite.tex_w, sprite.tex_h, sprite.slice_rows, sprite.slice_cols = width, height, rows, cols
    details._auto_slice_and_save(state, sprite)
    assert (state.settings.to_dict(), selection.snapshot, registry.require(state.document_id).revision) == before
    assert not history.action_journal.entries
    assert not registry.require(state.document_id).is_dirty
    assert not AssetImportProgressService.instance().is_active
    assert path.read_bytes() == b'source remains unchanged'
    assert meta.read_bytes() == b'authoring metadata remains unchanged'
    with pytest.raises(ValueError, match='grid'):
        details._auto_slice(state.settings, sprite)
    assert state.settings.to_dict() == before[0]


@pytest.mark.parametrize('width,height,rows,cols', [(8, 4, 2, 4), (7, 5, 2, 3), (4, 4, 4, 4), (1, 1, 1, 1)])
def test_valid_uniform_grid_keeps_positive_rectangles_and_stable_identities(width, height, rows, cols):
    settings = TextureImportSettings(texture_type=TextureType.SPRITE)
    sprite = details._SpriteEditorState()
    sprite.tex_w, sprite.tex_h, sprite.slice_rows, sprite.slice_cols = width, height, rows, cols
    details._auto_slice(settings, sprite)
    assert len(settings.sprite_frames) == rows * cols
    rectangles = {(f.x, f.y, f.w, f.h) for f in settings.sprite_frames}
    assert len(rectangles) == rows * cols
    for frame in settings.sprite_frames:
        assert frame.w == width // cols > 0 and frame.h == height // rows > 0
        assert 0 <= frame.x < frame.x + frame.w <= width
        assert 0 <= frame.y < frame.y + frame.h <= height
    settings.sprite_frames[0].name = 'Keep authored name'
    settings.sprite_frames[0].pivot_x = .25
    before = settings.to_dict()
    details._auto_slice(settings, sprite)
    assert settings.to_dict() == before
