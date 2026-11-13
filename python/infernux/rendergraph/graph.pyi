from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Mapping, Optional, Tuple, List, Dict

from infernux.lib import (
    RenderGraphDescription,
    GraphPassDesc,
    GraphTextureDesc,
    MaterialPassType,
    PixelFormat,
    DepthCompare as DepthCompare,
    InxTexture,
)
from infernux.renderstack.effect_stage import EffectScope, EffectStage
from infernux.renderstack.pass_result import PassResult
from infernux.core.render_texture import RenderTexture
from .renderer_selection import RendererSelection
from infernux.compute import Buffer, Work
from infernux.engine.cpu_continuation import CpuRequest
from infernux.execution import ResultConsumer

Format = PixelFormat


class TextureHandle:
    """A graph-local reference to a declared or imported texture resource."""

    name: str
    format: Format
    size: Optional[Tuple[int, int]]
    size_divisor: int
    samples: int
    asset_guid: str
    depth: int
    is_volume: bool

    def __init__(
        self,
        name: str,
        format: Format,
        is_camera_target: bool = ...,
        size: Optional[Tuple[int, int]] = ...,
        size_divisor: int = ...,
        samples: int = ...,
    ) -> None: ...
    @property
    def is_camera_target(self) -> bool:
        """Whether this resource uses a camera attachment."""
        ...
    @property
    def is_depth(self) -> bool:
        """Returns True if this texture uses a depth format."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


class BufferInput:
    """An imported read-only buffer slot that captures each published GPU value."""

    def __init__(self, initial: Buffer) -> None: ...
    def publish(self, value: Buffer) -> None: ...
    def require(self, request: CpuRequest) -> ResultConsumer: ...


class BufferHandle:
    """A handle to a buffer resource in the render graph."""

    name: str
    byte_size: int
    byte_offset: int
    usage: int
    view_light_list: bool

    def __init__(
        self,
        name: str,
        byte_size: int,
        usage: int,
        *,
        view_light_list: bool = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


class WorkBinding:
    """Graph-local ports; private storage belongs to each rendering view."""

    name: str
    def input(self, name: str) -> BufferHandle: ...
    def output(self, name: str) -> BufferHandle: ...


class RenderPassBuilder:
    """Fluent builder for constructing a render pass."""

    def __init__(self, name: str, graph: RenderGraph | None = ...) -> None: ...
    @property
    def name(self) -> str:
        """The name of this render pass."""
        ...
    def read(self, texture: str | TextureHandle) -> RenderPassBuilder:
        """Declare a texture as a read dependency for this pass."""
        ...
    def write_color(self, texture: str | TextureHandle, slot: int = ...) -> RenderPassBuilder:
        """Declare a color attachment output for this pass."""
        ...
    def write_depth(self, texture: str | TextureHandle) -> RenderPassBuilder:
        """Declare a depth attachment output for this pass."""
        ...
    def write_resolve(self, texture: str | TextureHandle) -> RenderPassBuilder:
        """Resolve color slot 0 into a single-sample texture."""
        ...
    def read_buffer(
        self, buffer: str | BufferHandle, usage: str = ...
    ) -> RenderPassBuilder:
        """Declare a storage, uniform, indirect, or transfer buffer read."""
        ...
    def write_buffer(
        self, buffer: str | BufferHandle, usage: str = ...
    ) -> RenderPassBuilder:
        """Declare a storage or transfer buffer write."""
        ...
    def set_side_effect(self, enabled: bool = ...) -> RenderPassBuilder:
        """Retain this pass for externally observable work."""
        ...
    def emit_event(self, label: str, *, buffers: Mapping[str, str | BufferHandle] | None = ...,
                   buffer: str | BufferHandle | None = ..., metadata: object = ...) -> RenderPassBuilder:
        """Declare an event for each accepted execution of this pass."""
        ...
    def set_texture(self, sampler_name: str, texture: str | TextureHandle) -> RenderPassBuilder:
        """Bind a texture to a sampler input for this pass."""
        ...
    def set_buffer(self, resource_name: str, buffer: str | BufferHandle) -> RenderPassBuilder:
        """Bind a read-only graph buffer to a fullscreen BufferFloat or BufferUInt resource."""
        ...
    def set_textures(self, bindings: Mapping[str, object]) -> RenderPassBuilder:
        """Bind multiple textures to sampler inputs for this pass."""
        ...
    def set_clear(
        self,
        color: Optional[Tuple[float, float, float, float]] = ...,
        depth: Optional[float] = ...,
    ) -> RenderPassBuilder:
        """Set clear values for color and/or depth attachments."""
        ...
    def draw_renderers(
        self,
        queue_range: Tuple[int, int] = ...,
        sort_mode: str = ...,
        pass_tag: str = ...,
        override_material: str = ...,
        material_pass: str = ...,
        material_filter: str = ...,
        renderer_selection: RendererSelection | None = ...,
    ) -> RenderPassBuilder:
        """Draw visible renderers filtered by queue range."""
        ...
    def draw_skybox(self) -> RenderPassBuilder:
        """Draw the skybox in this pass."""
        ...
    def draw_shadow_casters(
        self,
        queue_range: Tuple[int, int] = ...,
        light_index: int = ...,
    ) -> RenderPassBuilder:
        """Draw shadow-casting geometry for a light."""
        ...
    def draw_screen_ui(
        self,
        list: str = ...,
    ) -> RenderPassBuilder:
        """Draw screen-space UI elements in this pass."""
        ...
    def draw_world_ui(self, *, layer_mask: int = ...) -> RenderPassBuilder:
        """Draw selected GameObject layers, also respecting Camera culling and pass depth."""
        ...
    def fullscreen_quad(
        self,
        shader: str,
        *,
        depth_test: DepthCompare | None = None,
        depth_write: bool = False,
        alpha_blend: bool = False,
    ) -> RenderPassBuilder:
        """Draw a fullscreen triangle with explicit optional depth/blend state."""
        ...
    def copy_texture(
        self, source: str | TextureHandle, destination: str | TextureHandle
    ) -> RenderPassBuilder:
        """Copy one graph texture into another in a copy pass."""
        ...
    def copy_buffer(
        self,
        source: str | BufferHandle,
        destination: str | BufferHandle,
        byte_count: int = ...,
    ) -> RenderPassBuilder:
        """Copy bytes between graph buffers in a copy pass."""
        ...
    def present(self, source: str | TextureHandle) -> RenderPassBuilder:
        """Export a graph texture from a present pass."""
        ...
    def copy_texture_to_buffer(
        self, source: str | TextureHandle, destination: str | BufferHandle,
        *, size: Tuple[int, int], buffer_offset: int = 0,
    ) -> RenderPassBuilder:
        """Explicit GPU copy of a single-sample RGBA32F base-level rectangle."""
        ...
    def copy_buffer_to_texture(
        self, source: str | BufferHandle, destination: str | TextureHandle,
        *, buffer_offset: int = 0,
    ) -> RenderPassBuilder:
        """Fill a single-sample RGBA32F texture from packed buffer bytes."""
        ...
    def compute_work(self, binding: WorkBinding) -> RenderPassBuilder: ...
    def set_param(self, name: str, value: float) -> RenderPassBuilder:
        """Set a push-constant parameter for this pass."""
        ...
    def __enter__(self) -> RenderPassBuilder: ...
    def __exit__(self, *args: object) -> None: ...
    def __repr__(self) -> str: ...


class RenderGraph:
    """A declarative render graph that defines texture resources and render passes."""

    def __init__(self, name: str = ..., *, output_samples: int = 0) -> None: ...
    @property
    def name(self) -> str:
        """The name of this render graph."""
        ...
    @property
    def pass_count(self) -> int:
        """Number of render passes in the graph."""
        ...
    @property
    def texture_count(self) -> int:
        """Number of texture resources in the graph."""
        ...
    @property
    def buffer_count(self) -> int:
        """Number of buffer resources in the graph."""
        ...
    @property
    def topology_sequence(self) -> List[Tuple[str, str]]:
        """Ordered list of (pass_name, type) entries defining the execution order."""
        ...
    @property
    def injection_points(self) -> list:
        """List of injection points for pass extension."""
        ...
    @property
    def effect_stages(self) -> List[EffectStage]:
        """Pipeline-declared user attachment stages in topology order."""
        ...
    def set_temporal_jitter(self, enabled: bool = True) -> None: ...
    def set_msaa_samples(self, samples: int) -> int:
        """Set screen MSAA preference; return the effective Camera target sample count."""
        ...
    def create_texture(
        self,
        name: str,
        *,
        format: Format = ...,
        camera_target: bool = ...,
        size: Optional[Tuple[int, int]] = ...,
        size_divisor: int = ...,
        samples: Optional[int] = ...,
    ) -> TextureHandle:
        """Declare a transient texture resource in the render graph."""
        ...
    def get_texture(self, name: str) -> Optional[TextureHandle]:
        """Get a texture handle by name, or None if not found."""
        ...
    def import_texture(self, name: str, texture: RenderTexture | InxTexture, *, attachment: str = "color", require_current_write: bool | None = None) -> TextureHandle:
        """Import a persistent target attachment or a GUID-backed sample-only Texture asset."""
        ...
    def create_temporal_history(self, name: str, *, format: Format = Format.RGBA16_SFLOAT,
                                size: Optional[Tuple[int, int]] = None,
                                size_divisor: int = 0) -> Tuple[TextureHandle, TextureHandle]: ...
    def name_scope(self, prefix: str) -> AbstractContextManager[RenderGraph]: ...
    def effect_resources(
        self, resources: Mapping[str, TextureHandle]
    ) -> AbstractContextManager[RenderGraph]: ...
    @property
    def current_effect_resources(self) -> Mapping[str, TextureHandle]: ...
    def pass_result(self, result: PassResult) -> AbstractContextManager[RenderGraph]: ...
    @property
    def current_pass_result(self) -> PassResult | None: ...
    def replace_current_pass_result(self, result: PassResult) -> None: ...
    def resolve_effect_route_policy(self, stages): ...
    def create_buffer(
        self,
        name: str,
        byte_size: int,
        *,
        storage: bool = ...,
        indirect: bool = ...,
        transfer_source: bool = ...,
        transfer_destination: bool = ...,
    ) -> BufferHandle:
        """Declare a transient buffer resource in the render graph."""
        ...
    def create_view_light_list(self, name: str = "light_list") -> BufferHandle: ...
    def export_buffer(self, buffer: str | BufferHandle, *, shape=None, dtype=None) -> BufferHandle:
        """Expose this execution's complete final buffer as a captured GPU value."""
        ...
    def bind_work(
        self, name: str, work: Work, *, inputs: Optional[Mapping[str, BufferHandle]] = ...,
    ) -> WorkBinding:
        """Bind a common recipe without executing it or publishing ready data."""
        ...
    def import_buffer(self, name: str, buffer: Buffer | BufferInput) -> BufferHandle:
        """Import an accepted 32-bit scalar/vector GPU Buffer view without a hidden copy."""
        ...
    def get_buffer(self, name: str) -> Optional[BufferHandle]:
        """Get a buffer handle by name, or None if not found."""
        ...
    def has_pass(self, name: str) -> bool:
        """Check if a render pass with the given name exists."""
        ...
    def has_injection_point(self, name: str) -> bool:
        """Check if an injection point with the given name exists."""
        ...
    def has_effect_stage(self, stable_id: str) -> bool:
        """Check for an exact declared stage ID."""
        ...
    def injection_point(
        self,
        name: str,
        *,
        display_name: str = ...,
        resources: Optional[set] = ...,
    ) -> None:
        """Declare an injection point where external passes can be inserted."""
        ...
    def effect_stage(
        self,
        stable_id: str,
        *,
        scope: EffectScope | str = ...,
        display_name: str = ...,
        inputs: Optional[set[str]] = ...,
        outputs: Optional[set[str]] = ...,
        capabilities: Optional[set[str]] = ...,
    ) -> EffectStage:
        """Declare a stable user-facing RenderEffect attachment stage."""
        ...
    def effects(self, stable_id: str, **kwargs: object) -> EffectStage:
        """Pipeline-author shorthand for ``effect_stage``."""
        ...
    def screen_ui_section(self, *, resources: set | None = ..., world_ui_layer_mask: int = ...) -> None:
        """Declare a screen UI section in the graph topology."""
        ...
    def set_geometry_buffer_requirements(self, requirements) -> None: ...
    def require_geometry_buffers(self, requirements) -> None: ...
    @property
    def geometry_buffer_requirements(self): ...
    def needs_geometry_buffer(self, semantic: str) -> bool: ...
    def publish_pass_result(self, source: str, buffers, *, materialize=...) -> PassResult: ...
    def derive_pass_result(self, source: str, parent: PassResult, overrides) -> PassResult: ...
    def write_buffer(self, source: str, parent: PassResult, name: str, texture: TextureHandle) -> PassResult: ...
    @property
    def pass_results(self) -> Mapping[str, PassResult]: ...
    @property
    def latest_pass_result(self) -> PassResult | None: ...
    def get_pass_result(self, source: str) -> PassResult | None: ...
    def camera_ui_section(self, *, resources: set | None = ..., world_ui_layer_mask: int = ...) -> None:
        """Draw Camera UI and declare the after-camera-UI effect stage."""
        ...
    def screen_ui_overlay_section(self, *, resources: set | None = ...) -> None:
        """Encode for display, draw Screen UI, and declare its effect stage."""
        ...
    def add_pass(self, name: str) -> RenderPassBuilder:
        """Add a new render pass to the graph."""
        ...
    def add_copy_pass(self, name: str) -> RenderPassBuilder:
        """Add a transfer-domain texture or buffer copy pass."""
        ...
    def add_compute_pass(self, name: str) -> RenderPassBuilder: ...
    def add_present_pass(self, name: str) -> RenderPassBuilder:
        """Add a final graph export pass."""
        ...
    def remove_pass(self, name: str) -> RenderPassBuilder | None:
        """Remove a render pass by name. Returns the removed builder, or None."""
        ...
    def append_pass(self, builder: RenderPassBuilder) -> None:
        """Append an existing RenderPassBuilder to the graph."""
        ...
    def set_output(self, texture: str | TextureHandle) -> None:
        """Set the final output texture of the render graph."""
        ...
    def validate_no_ip_before_first_pass(self) -> None:
        """Validate that no user extension point appears before the first pass."""
        ...
    def get_debug_string(self) -> str:
        """Return a human-readable summary of the graph for debugging."""
        ...
    def build(self) -> RenderGraphDescription:
        """Compile the graph into a RenderGraphDescription for the backend."""
        ...
