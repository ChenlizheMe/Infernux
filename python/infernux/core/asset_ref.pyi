"""Type stubs for infernux.core.asset_ref."""

from __future__ import annotations

from typing import Any, Optional

from infernux.core.material import Material
from infernux.core.physic_material import PhysicMaterial
from infernux.core.render_texture import RenderTexture
from infernux.core.animation_clip import AnimationClip
from infernux.core.animation_clip3d import AnimationClip3D
from infernux.core.animation_timeline import AnimationTimeline
from infernux.core.anim_state_machine import AnimStateMachine
from infernux.particle.asset import ParticleGraphAsset
from infernux.lib import InxMaterial


class AssetRefBase:
    """Base class for GUID-based asset references."""

    def __init__(self, guid: str = ..., path_hint: str = ...) -> None:
        """Create an asset reference with an optional GUID and path hint."""
        ...
    @property
    def guid(self) -> str:
        """The globally unique identifier of the referenced asset."""
        ...
    @guid.setter
    def guid(self, value: str) -> None: ...
    @property
    def path_hint(self) -> str:
        """A file path hint for locating the asset."""
        ...
    @path_hint.setter
    def path_hint(self, value: str) -> None: ...
    def resolve(self) -> Optional[Any]:
        """Resolve the reference and return the loaded asset."""
        ...
    def invalidate(self) -> None:
        """Clear the cached resolved asset."""
        ...
    def to_dict(self) -> dict:
        """Serialize the reference to a dictionary."""
        ...
    @classmethod
    def from_dict(cls, d: dict) -> AssetRefBase:
        """Create an asset reference from a serialized dictionary."""
        ...
    @property
    def display_name(self) -> str:
        """A human-readable name for the referenced asset."""
        ...
    @property
    def is_missing(self) -> bool:
        """Whether the referenced asset cannot be found."""
        ...
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...


class GenericAssetRef(AssetRefBase):
    """Typed reference for registered assets without a specialized wrapper."""
    def __init__(
        self, asset_type: str, guid: str = ..., path_hint: str = ...
    ) -> None: ...
    @property
    def asset_type(self) -> str: ...


class TextureRef(AssetRefBase):
    """Reference to a Texture asset."""
    ...


class RenderTextureRef(AssetRefBase):
    """Reference to an imported RenderTexture description."""
    def resolve(self) -> RenderTexture | None: ...


class PhysicMaterialRef(AssetRefBase):
    """Reference to the shared Python PhysicMaterial proxy."""
    def resolve(self) -> PhysicMaterial | None: ...


class ShaderRef(AssetRefBase):
    """Reference to a Shader asset."""
    ...


class AudioClipRef(AssetRefBase):
    """Reference to an AudioClip asset."""
    ...


class DataAssetRef(AssetRefBase):
    """Reference to one shared typed DataAsset."""
    ...


class AnimationClipRef(AssetRefBase):
    def resolve(self) -> AnimationClip | None: ...


class AnimationClip3DRef(AssetRefBase):
    def resolve(self) -> AnimationClip3D | None: ...


class AnimationTimelineRef(AssetRefBase):
    def resolve(self) -> AnimationTimeline | None: ...


class AnimStateMachineRef(AssetRefBase):
    def resolve(self) -> AnimStateMachine | None: ...


class TimelineFSMRef(AssetRefBase):
    def resolve(self) -> AnimStateMachine | None: ...


class ParticleGraphRef(AssetRefBase):
    def resolve(self) -> ParticleGraphAsset | None: ...


class RenderEffectRef(AssetRefBase):
    """Reference to a mutable RenderEffect asset."""
    def __init__(self, effect: Any = ..., *, guid: str = ..., path_hint: str = ...) -> None: ...
    def resolve(self) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __copy__(self) -> RenderEffectRef: ...
    def __deepcopy__(self, memo: Any) -> RenderEffectRef: ...


class MaterialRef(AssetRefBase):
    """GUID-based reference to a Material asset."""

    def __init__(
        self, material: Material | InxMaterial | str | None = ..., *, guid: str = ..., path_hint: str = ...
    ) -> None: ...
    def resolve(self) -> Material | None:
        """Return the canonical Python Material proxy, or None if missing."""
        ...
    def __getattr__(self, name: str) -> Any: ...
    def __copy__(self) -> MaterialRef: ...
    def __deepcopy__(self, memo: Any) -> MaterialRef: ...
    def __eq__(self, other: object) -> bool: ...


def create_asset_ref(
    asset_type: str,
    *,
    guid: str = ...,
    path_hint: str = ...,
) -> AssetRefBase: ...

def get_asset_type_for_ref(ref: AssetRefBase | type[AssetRefBase]) -> Optional[str]: ...

def register_asset_type(
    asset_type: str, *, ref_class: type[AssetRefBase], drag_type: str,
    extensions: tuple[str, ...], display: str, prefix: str,
) -> None: ...

def get_asset_type_config(asset_type: str) -> Optional[dict]: ...
