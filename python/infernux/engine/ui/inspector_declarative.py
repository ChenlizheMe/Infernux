"""Declarative controls rendered by the common component Inspector path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from infernux.lib import InxGUIContext
from infernux.engine.i18n import t

from .inspector_utils import (
    field_label,
    has_field_changed,
    max_label_w,
    pretty_field_name,
    record_inspector_component_item,
    render_compact_section_header,
    render_serialized_field,
    render_section_kicker,
    serialized_field_label,
)
from .theme import ImGuiCol, ImGuiStyleVar, ImGuiTreeNodeFlags, Theme


@dataclass(frozen=True)
class InspectorChoice:
    key: str
    label: str
    options: Callable[[], Sequence[str]]
    current_index: Callable[[], int]
    on_change: Callable[[int], None]


@dataclass(frozen=True)
class InspectorSerializedTarget:
    key: str
    target: Callable[[], Any]
    owner: Any
    title: str = ""
    on_change: Callable[[Any, str, Any, Any], None] | None = None


@dataclass(frozen=True)
class InspectorReadOnlyRow:
    key: str
    label: str
    level: str = "tertiary"


@dataclass(frozen=True)
class InspectorList:
    key: str
    label: str
    target: Any
    field_name: str
    metadata: Any
    value: Callable[[], list]
    accept_drop: str | None = None
    drop_factory: Callable[[Any], Any] | None = None
    item_label: Callable[[Any, int], str] | None = None
    item_renderer: Callable[[InxGUIContext, Any, int, str], None] | None = None
    item_icon: str = ""
    item_domain: str = ""
    item_toggle: str = ""
    on_change: Callable[[Any, str, list, list], None] | None = None


@dataclass(frozen=True)
class InspectorInlineAssets:
    key: str
    asset_type: str
    references: Callable[[], Iterable[Any]]


@dataclass(frozen=True)
class InspectorMessages:
    key: str
    title: str
    messages: Callable[[], Iterable[str]]
    warning: bool = False


InspectorControl = (
    InspectorChoice
    | InspectorSerializedTarget
    | InspectorReadOnlyRow
    | InspectorList
    | InspectorInlineAssets
    | InspectorMessages
)


@dataclass(frozen=True)
class InspectorSection:
    key: str
    controls: tuple[InspectorControl, ...] = field(default_factory=tuple)
    title: str = ""
    level: str = "secondary"
    separator_before: bool = False


@dataclass(frozen=True)
class InspectorModel:
    key: str
    sections: tuple[InspectorSection, ...]


_INLINE_ASSET_RENDERERS: dict[str, Callable[[InxGUIContext, Iterable[Any], str], None]] = {}
_MODEL_PROVIDERS: dict[str, Callable[[Any], InspectorModel]] = {}
_MODEL_PRESENTERS: dict[str, Callable] = {}


def register_inline_asset_renderer(
    asset_type: str,
    renderer: Callable[[InxGUIContext, Iterable[Any], str], None],
) -> None:
    """Register a resource editor used by any declarative component model."""
    _INLINE_ASSET_RENDERERS[str(asset_type)] = renderer


def register_inspector_model_provider(
    component_type: str,
    provider: Callable[[Any], InspectorModel],
    *,
    presenter: Callable | None = None,
) -> None:
    """Register a data-model provider without giving the component draw code."""
    _MODEL_PROVIDERS[str(component_type)] = provider
    if presenter is not None:
        _MODEL_PRESENTERS[str(component_type)] = presenter


def get_inspector_model(component) -> InspectorModel | None:
    provider = _MODEL_PROVIDERS.get(str(getattr(component, "type_name", type(component).__name__)))
    return provider(component) if provider is not None else None


def render_inspector_model(ctx: InxGUIContext, component, model: InspectorModel) -> bool:
    """Render one declarative model. Returns True when structure changed."""
    structure_changed = False
    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)
    try:
        presenter = _MODEL_PRESENTERS.get(str(component.type_name))
        if presenter is not None:
            return bool(presenter(ctx, component, model))
        for section in model.sections:
            if section.separator_before:
                ctx.separator()
            if section.title and not render_compact_section_header(
                ctx,
                f"{section.title}##{model.key}_{section.key}",
                level=section.level,
            ):
                continue
            for control in section.controls:
                if isinstance(control, InspectorChoice):
                    options = tuple(control.options())
                    if not options:
                        continue
                    current = max(0, min(int(control.current_index()), len(options) - 1))
                    lw = max_label_w(ctx, [control.label])
                    field_label(ctx, control.label, lw)
                    selected = ctx.combo(f"##{model.key}_{control.key}", current, list(options), -1)
                    if selected != current:
                        control.on_change(int(selected))
                        structure_changed = True
                        return structure_changed
                elif isinstance(control, InspectorSerializedTarget):
                    _render_serialized_target(ctx, control)
                elif isinstance(control, InspectorReadOnlyRow):
                    _render_read_only_row(ctx, control, model.key)
                elif isinstance(control, InspectorList):
                    from ._inspector_list_field import _render_list_field

                    # Different declarative controls may expose the same
                    # property name (RenderStack stages all expose ``slots``).
                    # Scope nested widgets and picker popups to this control.
                    ctx.push_id_str(control.key)
                    try:
                        _render_list_field(
                            ctx,
                            control.target,
                            control.field_name,
                            control.metadata,
                            control.value(),
                            0.0,
                            display_name=control.label,
                            header_drop_type=control.accept_drop,
                            header_drop_factory=control.drop_factory,
                            item_label=control.item_label,
                            item_renderer=control.item_renderer,
                            item_icon=control.item_icon,
                            item_domain=control.item_domain,
                            item_toggle=control.item_toggle,
                            on_change=control.on_change,
                        )
                    finally:
                        ctx.pop_id()
                elif isinstance(control, InspectorInlineAssets):
                    renderer = _INLINE_ASSET_RENDERERS.get(control.asset_type)
                    if renderer is not None:
                        renderer(ctx, control.references(), f"{model.key}_{control.key}")
                elif isinstance(control, InspectorMessages):
                    _render_messages(ctx, control, model.key)
    finally:
        ctx.pop_style_var(2)
    return structure_changed


def _render_serialized_target(ctx: InxGUIContext, control: InspectorSerializedTarget) -> None:
    from infernux.components.fields import get_serialized_fields

    target = control.target()
    if target is None:
        return
    fields = get_serialized_fields(type(target))
    if not fields:
        return
    if control.title:
        render_section_kicker(ctx, control.title)
    labels = [serialized_field_label(name, meta) for name, meta in fields.items() if not meta.hidden]
    label_width = max_label_w(ctx, labels) if labels else 0.0
    current_group = ""
    current_header = ""
    group_visible = True
    for field_name, metadata in fields.items():
        if metadata.hidden:
            continue
        field_group = metadata.group or ""
        if field_group != current_group:
            current_group = field_group
            group_visible = (
                render_compact_section_header(
                    ctx, t(field_group), level="tertiary", header_id=f"{control.key}.group.{field_name}",
                )
                if field_group else True
            )
        if not group_visible:
            continue
        if metadata.visible_when is not None and not metadata.visible_when(target):
            continue
        current = getattr(target, field_name, metadata.default)
        display_name = serialized_field_label(field_name, metadata)

        def commit(updated, *, _name=field_name, _current=current):
            control.on_change(target, _name, _current, updated)

        from .field_draw import draw_field
        draw_field(ctx, metadata, current, label_width=label_width,
                   widget_id=f"##{control.key}_{field_name}", on_change=commit,
                   show_header=metadata.header != current_header)
        if metadata.header:
            current_header = metadata.header
        record_inspector_component_item(
            ctx, control.owner, f"{control.key}.{field_name}",
            "inspector_field", display_name, enabled=not metadata.readonly,
        )


def _render_read_only_row(ctx: InxGUIContext, control: InspectorReadOnlyRow, model_key: str) -> None:
    level = control.level
    if level == "primary":
        base = Theme.INSPECTOR_HEADER_PRIMARY
        hovered = Theme.INSPECTOR_HEADER_PRIMARY_HOVERED
        active = Theme.INSPECTOR_HEADER_PRIMARY_ACTIVE
    elif level == "list":
        base = Theme.INSPECTOR_HEADER_LIST
        hovered = Theme.INSPECTOR_HEADER_LIST_HOVERED
        active = Theme.INSPECTOR_HEADER_LIST_ACTIVE
    else:
        base = Theme.INSPECTOR_HEADER_TERTIARY
        hovered = Theme.INSPECTOR_HEADER_TERTIARY_HOVERED
        active = Theme.INSPECTOR_HEADER_TERTIARY_ACTIVE
    ctx.push_style_color(ImGuiCol.Header, *base)
    ctx.push_style_color(ImGuiCol.HeaderHovered, *hovered)
    ctx.push_style_color(ImGuiCol.HeaderActive, *active)
    ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
    try:
        ctx.tree_node_ex(
            f"{control.label}##{model_key}_{control.key}",
            ImGuiTreeNodeFlags.NoTreePushOnOpen
            | ImGuiTreeNodeFlags.Leaf
            | ImGuiTreeNodeFlags.Bullet
            | ImGuiTreeNodeFlags.SpanAvailWidth,
        )
    finally:
        ctx.pop_style_color(4)


def _render_messages(ctx: InxGUIContext, control: InspectorMessages, model_key: str) -> None:
    messages = tuple(str(message) for message in control.messages() if str(message))
    if not messages:
        return
    color = Theme.WARNING_TEXT if control.warning else Theme.META_TEXT
    if not render_compact_section_header(
        ctx,
        f"{control.title} [{len(messages)}]##{model_key}_{control.key}",
        level="secondary",
        text_color=color,
    ):
        return
    for message in messages:
        ctx.text_wrapped(message)
