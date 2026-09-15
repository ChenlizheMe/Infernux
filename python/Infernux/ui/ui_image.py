"""UIImage — a rectangular image UI element.

Hierarchy:
    InxComponent → InxUIComponent → InxUIScreenComponent → UIImage
"""

from Infernux.components import serialized_field, add_component_menu
from Infernux.components.fields import FieldType, FieldMetadata, SerializedFieldDescriptor
from .inx_ui_screen_component import InxUIScreenComponent


class _ImageTextureField(SerializedFieldDescriptor):
    """One authored asset slot with a non-serialized live target override."""

    def __init__(self):
        super().__init__(FieldMetadata(
            name="texture", default=None, field_type=FieldType.ASSET,
            asset_type="Texture.Sampled", group="Fill",
            tooltip="Texture or RenderTexture asset (drag from Project panel)",
        ))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        runtime = getattr(instance, '_render_texture', None)
        return runtime if runtime is not None else super().__get__(instance, owner)

    def __set__(self, instance, value):
        from Infernux.core.render_texture import RenderTexture
        if isinstance(value, RenderTexture) and not value.guid:
            instance._render_texture = value
            self._notify_runtime_value_changed(instance)
            return
        if (value is None and getattr(instance, '_render_texture', None) is not None
                and not getattr(instance, '_inf_deserializing', False)):
            instance._render_texture = None
            self._notify_runtime_value_changed(instance)
            return
        super().__set__(instance, value)
        instance._render_texture = None


@add_component_menu("UI/Image")
class UIImage(InxUIScreenComponent):
    """Screen or world image using a texture asset or live RenderTexture.

    Inherits x, y, width, height, opacity, corner_radius, rotation,
    mirror_x, mirror_y from InxUIScreenComponent.
    """

    # ── Fill ──
    texture = _ImageTextureField()
    color: list = serialized_field(
        default=[1.0, 1.0, 1.0, 1.0],
        field_type=FieldType.COLOR,
        hdr=True,
        tooltip="Tint color (RGBA)",
        group="Fill",
    )

    @property
    def texture_path(self):
        """Legacy authoring shorthand; only the texture GUID is serialized."""
        ref = type(self).texture.get_raw(self)
        return ref.path_hint if ref is not None else ""

    @texture_path.setter
    def texture_path(self, value):
        from Infernux.core.asset_ref import TextureRef
        self.texture = TextureRef(path_hint=value) if value else None

    def _image_texture_source(self):
        """Keep static image draws on the GPU asset cache, not CPU pixel loads."""
        from Infernux.core.asset_ref import RenderTextureRef
        runtime = getattr(self, '_render_texture', None)
        if runtime is not None:
            return runtime
        ref = type(self).texture.get_raw(self)
        return ref.resolve() if isinstance(ref, RenderTextureRef) else ref

    def _deserialize_fields_document(self, data, **kwargs):
        if isinstance(data, dict) and "texture_path" in data:
            from Infernux.core.asset_ref import TextureRef
            from Infernux.components.value_codec import VALUE_CODECS
            data = dict(data)
            legacy_path = data.pop("texture_path")
            if "texture" not in data:
                data["texture"] = (VALUE_CODECS.encode(TextureRef(path_hint=legacy_path))
                                   if legacy_path else None)
        super()._deserialize_fields_document(data, **kwargs)
