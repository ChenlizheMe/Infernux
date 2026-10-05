"""Effect-group Inspector handles leaf parameters and nested references."""

from types import SimpleNamespace

import pytest

from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.engine import interaction
from infernux.engine.ui import (
    _inspector_references,
    asset_details_renderer as details,
    inspector_utils,
    render_effect_inspector,
)
from infernux.renderstack.render_effect import EditableRenderEffectGroup, RenderEffect
from infernux.renderstack.render_effect_asset import (
    EffectAssetReference,
    RenderEffectAsset,
    RenderEffectGroupAsset,
    RenderEffectGroupEntry,
)


@pytest.mark.parametrize("nested", [False, True])
def test_group_inspector_renders_reference_and_only_leaf_parameters(monkeypatch, nested):
    guid = "referenced-guid"
    path = "referenced.effectgroup" if nested else "referenced.effect"
    resource = (
        EditableRenderEffectGroup(RenderEffectGroupAsset(), file_path=path, guid=guid)
        if nested
        else RenderEffect(
            RenderEffectAsset("test.inspector.feature", parameters={"intensity": 0.5}),
            file_path=path,
            guid=guid,
        )
    )
    group = EditableRenderEffectGroup(
        RenderEffectGroupAsset(
            entries=(RenderEffectGroupEntry("referenced", EffectAssetReference(guid=guid)),)
        ),
        file_path="root.effectgroup",
        guid="root-guid",
    )
    monkeypatch.setattr(AssetManager, "_get_path_from_guid", staticmethod(lambda value: path if value == guid else ""))
    monkeypatch.setattr(RenderEffectRef, "resolve", lambda self: resource)
    monkeypatch.setattr(inspector_utils, "render_compact_section_header", lambda *args, **kwargs: True)
    monkeypatch.setattr(details, "field_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(details, "max_label_w", lambda *args, **kwargs: 80)
    references = []
    monkeypatch.setattr(
        _inspector_references,
        "render_asset_reference_field",
        lambda *args, **kwargs: references.append(kwargs),
    )
    controllers = []
    flushed = []

    def ensure_controller(**kwargs):
        controllers.append(kwargs)
        return SimpleNamespace(document_id=kwargs["guid"], flush_autosave=lambda: flushed.append(kwargs["guid"]))

    monkeypatch.setattr(interaction, "ensure_editable_resource_document", ensure_controller)
    parameters = []
    monkeypatch.setattr(
        render_effect_inspector,
        "render_render_effect_parameters",
        lambda ctx, value, **kwargs: parameters.append((value, kwargs["resource_controller"])),
    )
    ctx = SimpleNamespace(
        label=lambda *args: None,
        separator=lambda: None,
        checkbox=lambda name, value: value,
        text_input=lambda name, value, size: value,
        button=lambda name: False,
        same_line=lambda: None,
        begin_disabled=lambda value: None,
        end_disabled=lambda: None,
    )
    state = SimpleNamespace(settings=group, resource_controller=None, extra={})

    details._render_render_effect_body(ctx, None, state)

    assert references[0]["semantic_id"] == "render_effect_group.entry.0.asset"
    assert references[0]["reference_value"]["guid"] == guid
    assert references[0]["ping_path"] == path
    assert references[-1]["semantic_id"] == "render_effect_group.add"
    if nested:
        assert not parameters and not controllers and not flushed
    else:
        assert len(parameters) == len(controllers) == 1
        assert parameters[0][0] is resource
        assert controllers[0]["resource"] is resource
        assert controllers[0]["guid"] == guid
        assert flushed == [guid]
