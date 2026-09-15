"""Persistent GPU render targets, owned independently of temporary graph handles.

Allocation, resize, Camera output, material/UI sampling and explicit RenderGraph
consumers are available in the graphical runtime. Imported descriptions are
resolved by GUID; runtime-created targets never fabricate an asset identity.
Unwritten pixels are undefined.
"""
from __future__ import annotations

import math

from Infernux.lib import PixelFormat, SampleCount


class RenderTexture:
    """A fixed or Game-relative GPU target with matching depth/MSAA attachments.

    Create on the engine thread, for example in ``start``. Resize publishes a
    complete allocation; recorded consumers retain their previous generation
    until GPU work completes. Dropping the last reference releases the target.
    No implicit pixel readback, disk cache, or secondary device is involved.

    Use ``RenderTexture(scale=(0.5, 0.5), ...)`` for half the Game render
    resolution. Do not provide width/height as well. Fractional pixels round
    up; actual ``width``/``height`` follow resolution changes automatically,
    independent of editor zoom, DPI, visibility and offscreen Camera order.
    Relative targets cannot be resized in pixels. Create a replacement to
    change the scale, just as when changing the target's format or MSAA.

    A pipeline imports attachments with ``graph.import_texture``. For MSAA,
    write the ``color`` and optional ``depth`` attachments, declare the
    ``resolve`` attachment with ``write_resolve``, then sample the resolve.
    Depth shader sampling is an explicit ``sampled_depth=True`` allocation
    use; ordinary depth testing does not require it.

    Assign a depth-equipped target to ``camera.target_texture`` to render that
    camera at the target's dimensions and sample count. Its output stays linear
    and excludes screen-space UI and display encoding. Assign ``None`` to
    return the camera to the screen output.

    An input-only ``graph.import_texture`` reads an active producer's output
    from the same frame, even when Camera depth orders the consumer first.
    Cycles, missing producers and same-target feedback are rejected; reading
    old pixels is not an implicit history mechanism.

    ``material.set_texture('texSampler', target)`` binds the sampled color
    (the resolved color for MSAA). The binding follows resize automatically
    and retains the resource. It does not alter a saved texture asset reference.
    Keep that material out of its producer camera's culling mask to avoid
    same-target feedback.

    ``image.texture = target`` displays the same resource in a UIImage, either
    on a screen canvas or as a world object. Assign ``None`` to restore the
    authored material/texture_path source. UI material texture bindings also
    work. World UI observes each camera's culling mask and depth buffer;
    screen Overlay encodes linear camera colors once for display.
    """

    def __init__(self, width: int | None = None, height: int | None = None, *,
                 scale: tuple[float, float] | None = None,
                 format: PixelFormat = PixelFormat.RGBA8_UNORM,
                 depth_format: PixelFormat = PixelFormat.UNDEFINED,
                 samples: int = 1, filter: str = "linear", storage: bool = False,
                 sampled_depth: bool = False):
        from Infernux.application import Application
        from Infernux.lib import _Infernux

        if scale is None:
            self._require_size(width, height)
        else:
            if width is not None or height is not None:
                raise ValueError("RenderTexture accepts pixel dimensions or scale, not both")
            if (not isinstance(scale, (tuple, list)) or len(scale) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in scale)):
                raise ValueError("RenderTexture scale must contain two positive finite numbers")
        if type(samples) is not int or samples not in (1, 2, 4, 8):
            raise ValueError("RenderTexture samples must be 1, 2, 4, or 8")
        if filter not in ("linear", "nearest"):
            raise ValueError("RenderTexture filter must be 'linear' or 'nearest'")
        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError("RenderTexture requires a running graphical engine")
        description = _Infernux._RenderTextureDesc()
        if scale is None:
            description.width, description.height = width, height
        else:
            description.relative_size = True
            description.width_scale, description.height_scale = scale
        description.color_format, description.depth_format = format, depth_format
        description.samples = SampleCount(samples)
        description.linear_filter = filter == "linear"
        description.storage = storage
        description.sampled_depth = sampled_depth
        self._native = engine.get_native_engine()._create_render_texture(description)

    @classmethod
    def load(cls, path: str) -> "RenderTexture":
        """Load an imported .rendertexture, including logical Player paths."""
        from Infernux.core.assets import AssetManager

        guid = AssetManager._get_guid_from_path(path)
        if not guid:
            raise FileNotFoundError(f"RenderTexture is not an imported asset: {path}")
        return cls.load_by_guid(guid)

    @classmethod
    def load_by_guid(cls, guid: str) -> "RenderTexture":
        """Resolve the shared GPU owner from its imported binary description."""
        from Infernux.application import Application

        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError("RenderTexture requires a running graphical engine")
        result = cls.__new__(cls)
        result._native = engine.get_native_engine()._load_render_texture(guid)
        return result

    @property
    def guid(self) -> str:
        """Imported asset identity; empty for a runtime-created target."""
        return self._native.asset_guid

    @property
    def file_path(self) -> str:
        from Infernux.core.assets import AssetManager

        return AssetManager._get_path_from_guid(self.guid) if self.guid else ""

    @property
    def display_name(self) -> str:
        """Editor asset label, without resolving or allocating another target."""
        from Infernux.core.asset_ref import RenderTextureRef

        return RenderTextureRef(self.guid).display_name if self.guid else "RenderTexture"

    @staticmethod
    def _require_size(width, height):
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("RenderTexture width and height must be positive integers")

    @property
    def width(self) -> int:
        return self._native.width

    @property
    def height(self) -> int:
        return self._native.height

    @property
    def scale(self) -> tuple[float, float] | None:
        """Game-relative scale, or None for a fixed-size target."""
        description = self._native.description
        if not description.relative_size:
            return None
        return description.width_scale, description.height_scale

    @property
    def format(self) -> PixelFormat:
        return self._native.description.color_format

    @property
    def depth_format(self) -> PixelFormat:
        return self._native.description.depth_format

    @property
    def samples(self) -> int:
        return int(self._native.description.samples)

    @property
    def sampled_depth(self) -> bool:
        """Whether the depth attachment supports explicit shader sampling."""
        return self._native.description.sampled_depth

    @property
    def revision(self) -> int:
        return self._native.revision

    @property
    def is_valid(self) -> bool:
        return self._native.is_valid

    @property
    def resident_bytes(self) -> int:
        """Attachment texel payload, excluding driver allocation padding."""
        return self._native.resident_bytes

    def resize(self, width: int, height: int) -> bool:
        """Change dimensions; return False if unchanged. Failure keeps ownership intact."""
        self._require_size(width, height)
        return self._native.resize(width, height)
