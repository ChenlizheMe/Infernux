from __future__ import annotations

import builtins

import pytest

from Infernux.engine.bootstrap_inspector._materials import (
    _rebuild_material_entries,
    wire_material_sections,
)


class _InspectorPanel:
    render_material_sections = None


class _Context:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def get_cursor_pos_y(self) -> float:
        return 0.0

    def text_wrapped(self, message: str) -> None:
        self.messages.append(message)


def test_material_section_import_failure_stays_inside_inspector(monkeypatch) -> None:
    panel = _InspectorPanel()
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "Infernux.engine.ui" and "inspector_material" in fromlist:
            raise ImportError("missing inspector_material test fixture")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    wire_material_sections(
        panel,
        None,
        object(),
        object(),
        lambda _object_id: (None, (), {}, {}),
        lambda _object_id: (None, 0, 0),
        {
            "object_id": 0,
            "scene_version": -1,
            "structure_version": -1,
            "signature": (),
            "entries": [],
        },
    )

    context = _Context()
    panel.render_material_sections(context, 42)

    assert context.messages == ["Material Inspector is temporarily unavailable."]


def test_material_slot_query_failure_reaches_inspector_boundary() -> None:
    class Renderer:
        type_name = "MeshRenderer"

        @staticmethod
        def get_effective_material(_slot):
            raise RuntimeError("invalid material binding")

    with pytest.raises(RuntimeError, match="invalid material binding"):
        _rebuild_material_entries([(Renderer(), 1, (), ())])


def test_texture_slot_uses_descriptor_drag_types_including_render_targets(monkeypatch):
    from types import SimpleNamespace
    from Infernux.engine.ui import inspector_material
    from Infernux.engine.ui.igui import IGUI
    from Infernux.engine.interaction.object_fields import AssetReferenceFieldModel

    monkeypatch.setattr(inspector_material, '_get_asset_database',
                        lambda: SimpleNamespace(get_path_from_guid=lambda guid: ''))
    monkeypatch.setattr(inspector_material, 'field_label', lambda *args: None)
    models = []

    def field(ctx, field_id, display_text, type_hint, **kwargs):
        models.append(AssetReferenceFieldModel(
            field_id=field_id, display_text=display_text, type_hint=type_hint, **kwargs))

    monkeypatch.setattr(IGUI, 'asset_reference_field', field)
    inspector_material._render_texture2d_property(object(), {'guid': ''}, 'texSampler', 'mat', 80)
    assert 'RENDER_TEXTURE_FILE' in models[0].accept
    assert 'TEXTURE_FILE' in models[0].accept


@pytest.mark.parametrize('extension,resource_type', [('png', 'Texture'), ('rendertexture', 'RenderTexture')])
def test_sampled_texture_clipboard_preserves_concrete_asset_type(extension, resource_type):
    from Infernux.core.asset_reference_types import AssetReferenceCodec, asset_type_registry

    encoded = AssetReferenceCodec.encode('Texture.Sampled', {'path_hint': f'Assets/Monitor.{extension}'})
    payload = AssetReferenceCodec.decode(encoded)
    assert payload['asset_type'] == resource_type
    assert not asset_type_registry.require(resource_type).incompatibility(payload)

