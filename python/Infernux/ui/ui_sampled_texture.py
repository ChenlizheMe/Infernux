"""Shared GUID-backed sampled texture slot for UI components."""

from Infernux.components.fields import FieldMetadata, FieldType, SerializedFieldDescriptor


class UISampledTextureField(SerializedFieldDescriptor):
    """One authored TextureRef/RenderTextureRef with a live RenderTexture override."""

    def __init__(self, *, name: str, tooltip: str):
        super().__init__(FieldMetadata(
            name=name,
            default=None,
            field_type=FieldType.ASSET,
            asset_type="Texture.Sampled",
            group="Fill",
            tooltip=tooltip,
        ))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        runtime = getattr(instance, "_render_texture", None)
        return runtime if runtime is not None else super().__get__(instance, owner)

    def __set__(self, instance, value):
        from Infernux.core.render_texture import RenderTexture

        if isinstance(value, RenderTexture) and not value.guid:
            instance._render_texture = value
            self._notify_runtime_value_changed(instance)
            return
        if (
            value is None
            and getattr(instance, "_render_texture", None) is not None
            and not getattr(instance, "_inf_deserializing", False)
        ):
            instance._render_texture = None
            self._notify_runtime_value_changed(instance)
            return
        super().__set__(instance, value)
        instance._render_texture = None


def ui_sampled_texture_source(instance, descriptor: UISampledTextureField):
    """Return the managed runtime target or current GUID-backed asset reference."""
    from Infernux.core.asset_ref import RenderTextureRef

    runtime = getattr(instance, "_render_texture", None)
    if runtime is not None:
        return runtime
    reference = descriptor.get_raw(instance)
    return reference.resolve() if isinstance(reference, RenderTextureRef) else reference
