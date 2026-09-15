"""CPU-only RenderTexture authoring using the shared resource document journal."""
from __future__ import annotations

import json
from pathlib import Path

from Infernux.lib import _Infernux as native


class RenderTextureDocument:
    """Editor document, not a second public GPU resource or resource cache."""

    def __init__(self, path: str):
        self.file_path = path
        self._description = native._render_texture_description_from_json(
            Path(path).read_text(encoding="utf-8"))

    def serialize_document(self) -> dict:
        return json.loads(native._render_texture_description_to_json(self._description))

    def deserialize_document(self, document: dict) -> None:
        # Parse first: a rejected authoring edit never changes the live document.
        self._description = native._render_texture_description_from_json(json.dumps(document))

    def save(self) -> None:
        from Infernux.core.assets import AssetManager
        from Infernux.core.document_store import write_document_text

        write_document_text(self.file_path,
                            native._render_texture_description_to_json(self._description) + "\n")
        result = AssetManager.reimport_asset(self.file_path)
        if not result:
            raise RuntimeError(result.error)


def load_document(path: str):
    return RenderTextureDocument(path), {}


def render_body(ctx, panel, state):
    from Infernux.engine.i18n import t
    from .asset_details_renderer import _apply_editable_resource_document
    from .inspector_utils import field_label, max_label_w, render_compact_section_header

    if not render_compact_section_header(ctx, t("asset.render_texture_properties"), level="secondary"):
        return
    document = state.settings.serialize_document()
    labels = {key: t(f"asset.rt_{key}") for key in (
        "size_mode", "width", "height", "width_scale", "height_scale",
        "format", "depth_format", "samples", "filter", "storage", "sampled_depth")}
    label_width = max_label_w(ctx, list(labels.values()))

    def combo(key, values, current, display=None):
        field_label(ctx, labels[key], label_width)
        return values[ctx.combo(f"##rt_{key}", values.index(current), display or values)]

    relative = "scale" in document["size"]
    next_relative = bool(combo("size_mode", [0, 1], int(relative),
                               [t("asset.rt_fixed"), t("asset.rt_relative")]))
    if next_relative != relative:
        document["size"] = {"scale": [1.0, 1.0]} if next_relative else {"width": 256, "height": 256}
    size = document["size"]
    if next_relative:
        for index, key in enumerate(("width_scale", "height_scale")):
            field_label(ctx, labels[key], label_width)
            size["scale"][index] = ctx.drag_float(f"##rt_{key}", size["scale"][index], 0.01, 0.001, 0.0)
            ctx.record_semantic_item("drag_float", labels[key], True, f"rt_{key}",
                                     numeric_value=size["scale"][index])
    else:
        for key in ("width", "height"):
            field_label(ctx, labels[key], label_width)
            size[key] = ctx.drag_int(f"##rt_{key}", size[key], 1.0, 1, 2147483647)
            ctx.record_semantic_item("drag_int", labels[key], True, f"rt_{key}", numeric_value=size[key])
    for key, depth in (("format", False), ("depth_format", True)):
        document[key] = combo(key, native._render_texture_format_names(depth), document[key])
    document["samples"] = combo("samples", [1, 2, 4, 8], document["samples"], ["1", "2", "4", "8"])
    document["filter"] = combo("filter", ["nearest", "linear"], document["filter"],
                               [t("asset.filter_point"), t("asset.filter_bilinear")])
    field_label(ctx, labels["storage"], label_width)
    document["storage"] = bool(ctx.checkbox("##rt_storage", document["storage"]))
    ctx.record_semantic_item("checkbox", labels["storage"], True, "rt_storage", bool_value=document["storage"])
    if document["depth_format"] != "undefined":
        field_label(ctx, labels["sampled_depth"], label_width)
        document["sampled_depth"] = bool(ctx.checkbox("##rt_sampled_depth", document["sampled_depth"]))
        ctx.record_semantic_item("checkbox", labels["sampled_depth"], True, "rt_sampled_depth",
                                 bool_value=document["sampled_depth"])
    else:
        document["sampled_depth"] = False
    if document != state.settings.serialize_document():
        _apply_editable_resource_document(state, document, edit_key="render_texture.description",
                                          description="Edit RenderTexture")
