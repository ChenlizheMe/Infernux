"""
RenderGraph builder API.

Pure Python graph builder that constructs a RenderGraphDescription.
The description is then sent to C++ for DAG compilation and execution.

Design: builder pattern with a fluent API for straightforward authoring.

    graph = RenderGraph("ForwardPipeline")
    graph.create_texture("color", camera_target=True)
    graph.create_texture("depth", format=Format.D32_SFLOAT)

    with graph.add_pass("Opaque") as p:
        p.write_color("color")
        p.write_depth("depth")
        p.set_clear(color=(0.1, 0.1, 0.1, 1.0), depth=1.0)
        p.draw_renderers(queue_range=(0, 2500), sort_mode="front_to_back")

    graph.set_output("color")
    desc = graph.build()  # -> RenderGraphDescription (C++ POD)
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from itertools import count
from operator import index
from typing import Mapping, Optional, Tuple, List, Dict
from .renderer_selection import RendererSelection

# Graph descriptions use the same native schema in tests, Editor and Player.
from Infernux.lib import (
    RenderGraphDescription,
    GraphPassDesc,
    GraphCommandDesc,
    GraphCommandType,
    GraphMaterialFilter,
    GraphPassType,
    GraphBufferDesc,
    GraphBufferUsage,
    GraphBufferAccessDesc,
    GraphBufferAccessType,
    GraphTextureDesc,
    GraphTextureRole,
    GraphTextureAttachment,
    MaterialPassType,
    PixelFormat,
    DepthCompare,
)
_SOURCE_REVISION_COUNTER = count(1)


# ============================================================================
# Public format type
# ============================================================================

Format = PixelFormat


# ============================================================================
# TextureHandle — lightweight handle to a graph texture resource
# ============================================================================

class TextureHandle:
    """Opaque handle to a texture resource in the RenderGraph.

    Not meant to be constructed directly — use ``RenderGraph.create_texture()``.
    """

    def __init__(self, name: str, format: Format, is_camera_target: bool = False,
                 size: "Optional[Tuple[int, int]]" = None,
                 size_divisor: int = 0,
                 samples: int = 1,
                 temporal_role=GraphTextureRole.TRANSIENT,
                 temporal_key: str = ""):
        self.name = name
        self.format = format
        self.is_camera_target = is_camera_target
        self.size = size  # (width, height) or None for scene target size
        self.size_divisor = size_divisor  # >1: scene_size / divisor
        self.samples = samples  # 0 inherits the graph's frame MSAA setting
        self.temporal_role = temporal_role
        self.temporal_key = temporal_key
        self.render_texture = None
        self.attachment = GraphTextureAttachment.COLOR
        self.asset_guid = ""
        self.depth = 1
        self.is_volume = False

    @property
    def is_depth(self) -> bool:
        return self.format.is_depth

    def __repr__(self) -> str:
        tag = " [camera_target]" if self.is_camera_target else ""
        return f"<TextureHandle '{self.name}' {self.format.name}{tag}>"

    def __eq__(self, other):
        if isinstance(other, TextureHandle):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


class BufferHandle:
    """Opaque handle to a graph buffer."""

    def __init__(self, name: str, byte_size: int, usage: int):
        self.name = name
        self.byte_size = byte_size
        self.usage = usage
        self.compute_buffer = None

    def __repr__(self) -> str:
        return f"<BufferHandle '{self.name}' {self.byte_size} bytes>"

    def __eq__(self, other):
        if isinstance(other, BufferHandle):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


# ============================================================================
# RenderPassBuilder — configures a single pass in the graph
# ============================================================================

class RenderPassBuilder:
    """Builder for configuring a single render pass.

    Provides a fluent API for declaring inputs, outputs, clear settings,
    and the render action for a pass. Also usable as a context manager::

        with graph.add_pass("OpaquePass") as p:
            p.write_color("color")
            p.draw_renderers(queue_range=(0, 2500))

    All resource arguments (``read``, ``write_color``, ``write_depth``,
    ``set_input``) accept **either** a string alias (resolved via
    ``graph.get_texture()``) or a ``TextureHandle`` directly.
    """

    def __init__(self, name: str, graph: "RenderGraph | None" = None,
                 pass_type: str = "raster"):
        self._name = name
        self._graph = graph
        self._pass_type = pass_type
        self._reads: List[str] = []
        self._buffer_accesses: List[Tuple[str, str]] = []
        self._write_colors: Dict[int, str] = {}  # slot -> texture_name (MRT)
        self._write_depth: Optional[str] = None
        self._resolve_color: Optional[str] = None
        self._clear_color: Optional[Tuple[float, float, float, float]] = None
        self._clear_depth: Optional[float] = None
        self._action = "none"
        self._material_pass = "forward"
        self._material_filter = "all"
        self._queue_min = 0
        self._queue_max = 5000
        self._sort_mode = "none"
        self._pass_tag = ""
        self._override_material = ""
        self._renderer_selection = None
        self._input_bindings: Dict[str, str] = {}  # shader resource -> graph resource
        self._light_index = 0
        self._screen_ui_list = 0
        self._world_ui_layer_mask = 0xffffffff
        self._shader_name: str = ""
        self._depth_test: Optional[DepthCompare] = None
        self._depth_write = False
        self._alpha_blend = False
        self._parameter_block: str = ""
        self._push_constants: Dict[str, float] = {}
        self._source_resource = ""
        self._destination_resource = ""
        self._copy_bytes = 0
        self._side_effect = False

    @property
    def name(self) -> str:
        return self._name

    # ---- String / handle resolution ----

    def _resolve(self, texture) -> "TextureHandle":
        """Resolve a string alias or ``TextureHandle`` to ``TextureHandle``."""
        if isinstance(texture, str):
            if self._graph is None:
                raise ValueError(
                    f"Cannot resolve alias '{texture}' without graph reference"
                )
            handle = self._graph.get_texture(texture)
            if handle is None:
                raise ValueError(
                    f"Texture '{texture}' not found in graph "
                    f"'{self._graph.name}'"
                )
            return handle
        if not isinstance(texture, TextureHandle):
            raise TypeError("Expected a texture alias or TextureHandle")
        if self._graph is None or not self._graph._owns_texture(texture):
            raise ValueError("TextureHandle does not belong to this RenderGraph")
        return texture

    def _resolve_buffer(self, buffer) -> "BufferHandle":
        if isinstance(buffer, str):
            if self._graph is None:
                raise ValueError(
                    f"Cannot resolve buffer alias '{buffer}' without graph reference"
                )
            handle = self._graph.get_buffer(buffer)
            if handle is None:
                raise ValueError(
                    f"Buffer '{buffer}' not found in graph '{self._graph.name}'"
                )
            return handle
        if not isinstance(buffer, BufferHandle):
            raise TypeError("Expected a buffer alias or BufferHandle")
        if self._graph is None or not self._graph._owns_buffer(buffer):
            raise ValueError("BufferHandle does not belong to this RenderGraph")
        return buffer

    # ---- Resource declarations ----

    def read(self, texture) -> "RenderPassBuilder":
        """Declare a texture input dependency.

        Args:
            texture: Texture alias (``str``) or ``TextureHandle``.
        """
        handle = self._resolve(texture)
        if handle.name not in self._reads:
            self._reads.append(handle.name)
        return self

    def write_color(self, texture, slot: int = 0) -> "RenderPassBuilder":
        """Declare a color output attachment.

        Args:
            texture: Texture alias (``str``) or ``TextureHandle``.
            slot: Color attachment slot (0 = primary, higher = MRT).
        """
        handle = self._resolve(texture)
        self._write_colors[slot] = handle.name
        return self

    def write_depth(self, texture) -> "RenderPassBuilder":
        """Declare the depth output attachment.

        Args:
            texture: Texture alias (``str``) or ``TextureHandle``.
        """
        handle = self._resolve(texture)
        self._write_depth = handle.name
        return self

    def write_resolve(self, texture) -> "RenderPassBuilder":
        """Resolve multisampled color slot 0 into a single-sample texture.

        The pass must declare exactly one multisampled color output.  The
        resolve target must have the same format and extent with ``samples=1``.
        """
        handle = self._resolve(texture)
        self._resolve_color = handle.name
        return self

    def read_buffer(self, buffer, usage: str = "storage") -> "RenderPassBuilder":
        """Declare a storage, indirect, or transfer buffer read."""
        handle = self._resolve_buffer(buffer)
        access = {
            "storage": "storage_read",
            "indirect": "indirect_read",
            "transfer": "transfer_read",
        }.get(str(usage).strip().lower())
        if access is None:
            raise ValueError(f"Unknown buffer read usage '{usage}'")
        self._buffer_accesses.append((handle.name, access))
        return self

    def write_buffer(self, buffer, usage: str = "storage") -> "RenderPassBuilder":
        """Declare a storage or transfer buffer write."""
        handle = self._resolve_buffer(buffer)
        if handle.compute_buffer is not None:
            raise ValueError(f"Imported buffer '{handle.name}' is read-only in the render graph")
        access = {
            "storage": "storage_write",
            "transfer": "transfer_write",
        }.get(str(usage).strip().lower())
        if access is None:
            raise ValueError(f"Unknown buffer write usage '{usage}'")
        self._buffer_accesses.append((handle.name, access))
        return self

    def set_side_effect(self, enabled: bool = True) -> "RenderPassBuilder":
        """Retain this pass for externally observable work."""
        self._side_effect = bool(enabled)
        return self

    def set_texture(
        self,
        sampler_name: str,
        texture,
    ) -> "RenderPassBuilder":
        """Bind a graph texture to a shader sampler.

        Fullscreen shaders declare ``Texture2DMS`` (or ``Texture2DMSUInt``)
        in ShaderInfo Resources to read individual MSAA samples with
        ``texelFetch``. These inputs retain the multisampled color/depth image;
        ordinary ``Texture2D`` inputs receive resolved data. A multisampled
        input cannot bind a single-sample texture. Use the effective count
        returned by ``set_msaa_samples`` when selecting a shader for a Camera
        with a fixed RenderTexture target. Categorical IDs must not be averaged.

        Args:
            sampler_name: Sampler name in the shader
                          (e.g. ``"shadowMap"``).
            texture: Texture alias (``str``) or ``TextureHandle``.
        """
        handle = self._resolve(texture)
        self._input_bindings[sampler_name] = handle.name
        if handle.name not in self._reads:
            self._reads.append(handle.name)
        return self

    def set_textures(
        self,
        bindings: Mapping[str, object],
    ) -> "RenderPassBuilder":
        """Bind multiple graph textures to shader samplers in one call.

        This removes repeated ``set_texture()`` boilerplate in multi-input
        passes such as deferred lighting and post-processing.

        Args:
            bindings: Mapping of ``sampler_name -> texture`` where *texture*
                is either a string alias or ``TextureHandle``.
        """
        for sampler_name, texture in bindings.items():
            self.set_texture(sampler_name, texture)
        return self

    def set_buffer(self, resource_name: str, buffer) -> "RenderPassBuilder":
        """Bind a read-only graph buffer to a fullscreen shader ``BufferUInt`` resource.

        Shader Resources and the pass's input calls use the same binding order.
        """
        handle = self._resolve_buffer(buffer)
        if resource_name in self._input_bindings:
            raise ValueError(f"Shader input '{resource_name}' is already bound")
        self._input_bindings[resource_name] = handle.name
        self.read_buffer(handle)
        return self

    # ---- Clear settings ----

    def set_clear(
        self,
        color: Optional[Tuple[float, float, float, float]] = None,
        depth: Optional[float] = None,
    ) -> "RenderPassBuilder":
        """Set clear values for this pass.

        Args:
            color: RGBA clear color tuple, or None to not clear color.
            depth: Depth clear value, or None to not clear depth.
        """
        self._clear_color = color
        self._clear_depth = depth
        return self

    # ---- Render actions ----

    def draw_renderers(
        self,
        queue_range: Tuple[int, int] = (0, 5000),
        sort_mode: str = "none",
        pass_tag: str = "",
        override_material: str = "",
        material_pass: str = "forward",
        material_filter: str = "all",
        renderer_selection: RendererSelection | None = None,
    ) -> "RenderPassBuilder":
        """Configure this pass to draw scene renderers.

        Args:
            queue_range: (min, max) inclusive render queue range for filtering.
                         Opaque = (0, 2500), Transparent = (2501, 5000).
            sort_mode: Sorting strategy — "front_to_back", "back_to_front",
                       or "none".
            pass_tag: Filter draw calls by shader pass tag (empty = no filter).
            override_material: Force accepted objects to use a built-in material
                key or a project material GUID.
                               (empty = per-object material).
            material_pass: Linked material program used by this pass. Supported
            values are ``forward``, ``forward_plus``, ``gbuffer``, ``depth``,
                           ``picking``, ``motion``, ``normal``, and ``base_color``.
            material_filter: Select all materials, only Deferred-compatible
                             materials, or only models that declare
                             ``Unsupported [Deferred]``.
            renderer_selection: Explicit Renderer/submesh selection with its
                own material and captured parameters. Intersects camera
                visibility and source queue/tag filters. Cannot be combined
                with ``override_material``.
        """
        if renderer_selection is not None:
            if not isinstance(renderer_selection, RendererSelection):
                raise TypeError("renderer_selection must be a RendererSelection")
            if override_material:
                raise ValueError("renderer_selection already supplies its material")
        normalized_pass = str(material_pass).strip().lower()
        if normalized_pass not in {
            "forward",
            "forward_plus",
            "gbuffer",
            "depth",
            "picking",
            "motion",
            "normal",
            "base_color",
        }:
            raise ValueError(f"Unknown material pass '{material_pass}'")
        self._action = "draw_renderers"
        self._material_pass = normalized_pass
        normalized_filter = str(material_filter).strip().lower()
        if normalized_filter not in {"all", "deferred_compatible", "deferred_unsupported"}:
            raise ValueError(f"Unknown material filter '{material_filter}'")
        self._material_filter = normalized_filter
        self._queue_min, self._queue_max = queue_range
        self._sort_mode = sort_mode
        self._pass_tag = pass_tag
        self._override_material = override_material
        self._renderer_selection = renderer_selection
        return self

    def draw_skybox(self) -> "RenderPassBuilder":
        """Configure this pass to draw the procedural skybox."""
        self._action = "draw_skybox"
        return self

    def draw_shadow_casters(
        self,
        queue_range: Tuple[int, int] = (0, 2999),
        light_index: int = 0,
    ) -> "RenderPassBuilder":
        """Configure this pass to render shadow casters into a depth-only shadow map.

        Each material uses its own vertex shader with a per-material shadow
        fragment variant (auto-generated).  Front-face culling and depth bias
        are applied for shadow acne prevention.

        Hard vs. soft shadows are NOT configured here: the shadow map only
        stores depth. Filtering (16-tap Vogel-disk PCF) is selected per light
        via ``light.shadows = LightShadows.Soft``.

        Args:
            queue_range: (min, max) inclusive render queue range for shadow casters.
                         Default (0, 2999) covers all opaque geometry regardless of queue.
            light_index: Index of the shadow-casting light (0 = first directional).
        """
        self._action = "draw_shadow_casters"
        self._material_pass = "shadow"
        self._queue_min, self._queue_max = queue_range
        self._light_index = light_index
        return self

    def draw_screen_ui(
        self,
        list: str = "camera",
    ) -> "RenderPassBuilder":
        """Configure this pass to draw screen-space UI.

        The runtime submission service accumulates UI commands through
        InxScreenUIRenderer at camera submission, then this pass renders the
        immutable command snapshot inside the scene render graph.

        Args:
            list: ``"camera"`` (before post-process, affected by post-processing)
                  or ``"overlay"`` (after post-process, on top of everything).
        """
        _str_to_int = {"camera": 0, "overlay": 1}
        value = _str_to_int.get(list.lower())
        if value is None:
            raise ValueError(
                f"Unknown screen UI list '{list}'. "
                f"Expected 'camera' or 'overlay'."
            )
        self._action = "draw_screen_ui"
        self._screen_ui_list = value
        return self

    def draw_world_ui(self, *, layer_mask: int = 0xffffffff) -> "RenderPassBuilder":
        """Draw world UI on the selected GameObject layers.

        The unsigned 32-bit mask is intersected with the Camera's culling mask.
        UI keeps its normal depth test against this pass's depth attachment.
        Disjoint masks can place ordinary UI and late labels in separate passes.
        """
        if isinstance(layer_mask, bool):
            raise TypeError("layer_mask must be an unsigned 32-bit integer")
        mask = index(layer_mask)
        if not 0 <= mask <= 0xffffffff:
            raise ValueError("layer_mask must be an unsigned 32-bit integer")
        self._action = "draw_world_ui"
        self._world_ui_layer_mask = mask
        return self

    def set_param(
        self,
        name: str,
        value: float,
    ) -> "RenderPassBuilder":
        """Set a named push-constant parameter for this pass.

        Parameters are passed to the fragment shader as push constants
        in the order they are declared.  Call once per parameter::

            p.set_param("intensity", 0.8)
            p.set_param("threshold", 1.0)

        Args:
            name: Parameter name (must match the shader push-constant
                  struct field name).
            value: Float value.
        """
        self._push_constants[name] = float(value)
        return self

    def bind_parameter_block(
        self,
        block_id: str,
        parameters: Mapping[str, float],
    ) -> "RenderPassBuilder":
        """Bind revisioned runtime values without making them graph topology.

        Ordered parameter names become part of the compiled command contract.
        Values may later be updated through the render context without
        rebuilding the graph.
        """
        normalized_id = str(block_id or "").strip()
        if not normalized_id:
            raise ValueError("parameter block id cannot be empty")
        if self._parameter_block and self._parameter_block != normalized_id:
            raise ValueError("a render command can bind only one parameter block")
        if not isinstance(parameters, Mapping):
            raise TypeError("parameter block values must be a mapping")
        self._parameter_block = normalized_id
        self._push_constants.clear()
        for name, value in parameters.items():
            parameter_name = str(name or "").strip()
            if not parameter_name:
                raise ValueError("parameter block names cannot be empty")
            self._push_constants[parameter_name] = float(value)
        if len(self._push_constants) > 32:
            raise ValueError("fullscreen parameter blocks support at most 32 floats")
        return self

    def fullscreen_quad(
        self,
        shader: str,
        *,
        depth_test: Optional[DepthCompare] = None,
        depth_write: bool = False,
        alpha_blend: bool = False,
    ) -> "RenderPassBuilder":
        """Configure this pass to draw a fullscreen triangle with a named shader.

        The vertex shader is always ``fullscreen_triangle``; the fragment
        shader is looked up by its matching ``ShaderInfo Name``.

        Use ``set_param()`` to pass push constants and ``set_texture()``
        to bind input textures before calling this method.

        Args:
            shader: Fragment shader id (e.g. ``"Bloom Prefilter"``).
            depth_test: Comparison against ``write_depth()``; None disables it.
            depth_write: Write shader/fixed fragment depth. Requires depth_test;
                use DepthCompare.ALWAYS for unconditional writes.
            alpha_blend: Straight-alpha source-over blending. Blending and
                depth-tested passes load previous color unless set_clear() is
                explicit. A new transient attachment must first be cleared.

        Sampled scene depth must be a separate resource from the attachment;
        copy it first if this pass also updates the scene depth.
        """
        if depth_test is not None and not isinstance(depth_test, DepthCompare):
            raise TypeError("depth_test must be a DepthCompare or None")
        if depth_write and depth_test is None:
            raise ValueError("depth_write requires depth_test (use DepthCompare.ALWAYS)")
        self._action = "fullscreen_quad"
        self._shader_name = shader
        self._depth_test = depth_test
        self._depth_write = bool(depth_write)
        self._alpha_blend = bool(alpha_blend)
        return self

    def copy_texture(self, source, destination) -> "RenderPassBuilder":
        """Copy one graph texture into another in a copy pass."""
        if self._pass_type != "copy":
            raise ValueError("copy_texture() requires graph.add_copy_pass()")
        self._action = "copy_texture"
        self._source_resource = self._resolve(source).name
        self._destination_resource = self._resolve(destination).name
        return self

    def copy_buffer(self, source, destination, byte_count: int = 0) -> "RenderPassBuilder":
        """Copy bytes between graph buffers in a copy pass."""
        if self._pass_type != "copy":
            raise ValueError("copy_buffer() requires graph.add_copy_pass()")
        if byte_count < 0:
            raise ValueError("byte_count must be >= 0")
        source_handle = self._resolve_buffer(source)
        destination_handle = self._resolve_buffer(destination)
        if destination_handle.compute_buffer is not None:
            raise ValueError(f"Imported buffer '{destination_handle.name}' cannot be a copy destination")
        source_handle.usage |= int(GraphBufferUsage.TRANSFER_SOURCE)
        destination_handle.usage |= int(GraphBufferUsage.TRANSFER_DESTINATION)
        self._action = "copy_buffer"
        self._source_resource = source_handle.name
        self._destination_resource = destination_handle.name
        self._copy_bytes = int(byte_count)
        return self

    def present(self, source) -> "RenderPassBuilder":
        """Export a graph texture from a present pass."""
        if self._pass_type != "present":
            raise ValueError("present() requires graph.add_present_pass()")
        handle = self._resolve(source)
        self._action = "present"
        self._source_resource = handle.name
        if self._graph is not None:
            self._graph.set_output(handle)
        return self

    # ---- Context manager support ----

    def __enter__(self) -> "RenderPassBuilder":
        return self

    def __exit__(self, *args):
        pass

    def __repr__(self) -> str:
        return (f"<RenderPassBuilder '{self._name}' "
                f"action={self._action}>")


# ============================================================================
# RenderGraph — the main graph builder
# ============================================================================

class RenderGraph:
    """Python-side RenderGraph topology builder.

    Unified API: textures have string aliases, injection points are
    declared inline, and the topology sequence is auto-recorded.

    Example::

        graph = RenderGraph("ForwardPipeline")

        graph.create_texture("color", camera_target=True)
        graph.create_texture("depth", format=Format.D32_SFLOAT)

        with graph.add_pass("OpaquePass") as p:
            p.write_color("color")
            p.write_depth("depth")
            p.set_clear(color=(0.1, 0.1, 0.1, 1.0), depth=1.0)
            p.draw_renderers(queue_range=(0, 2500), sort_mode="front_to_back")

        graph.injection_point("after_opaque", resources={"color", "depth"})

        graph.set_output("color")
        desc = graph.build()
    """

    def __init__(self, name: str = "RenderGraph", *, output_samples: int = 0):
        if output_samples not in (0, 1, 2, 4, 8):
            raise ValueError("output_samples must be 0, 1, 2, 4, or 8")
        self._output_samples = output_samples
        self._name = name
        self._textures: List[TextureHandle] = []
        self._buffers: List[BufferHandle] = []
        self._passes: List[RenderPassBuilder] = []
        self._output: Optional[str] = None
        self._linear_output: Optional[str] = None
        self._msaa_samples: int = output_samples  # 0 = pipeline-owned screen MSAA
        self._temporal_jitter = False
        # Topology auto-recording
        self._topology: List[Tuple[str, str]] = []
        self._injection_points_list: List = []  # List[InjectionPoint]
        self._effect_stages_list: List = []  # List[EffectStage]
        # Optional callback invoked at each injection_point() (set by RenderStack)
        self._injection_callback = None
        # Optional callback invoked at each pipeline-declared EffectStage.
        self._effect_stage_callback = None
        self._effect_route_policy_resolver = None
        self._effect_stage_active_resolver = None
        self._name_scopes: List[str] = []
        self._effect_resource_scopes: List[Dict[str, TextureHandle]] = []
        self._pass_result_scopes: List[object] = []
        self._geometry_buffer_requirements = frozenset()
        self._pass_results: Dict[str, object] = {}
        self._pass_result_revision = 0

    @property
    def name(self) -> str:
        return self._name

    def set_temporal_jitter(self, enabled: bool = True) -> None:
        """Request per-view camera jitter (for TAA), not implicit in history allocation."""
        self._temporal_jitter = bool(enabled)

    def set_msaa_samples(self, samples: int) -> int:
        """Set MSAA sample count for this graph (1=off, 2, 4, 8).

        Return the effective sample count used to construct attachments and
        resolve passes. A fixed Camera target overrides the pipeline's screen
        preference. Use the returned value for topology dependent on MSAA;
        authored pipeline parameters are never modified per camera.
        """
        if samples not in (0, 1, 2, 4, 8):
            raise ValueError(f"Invalid MSAA sample count: {samples}. Must be 0, 1, 2, 4, or 8.")
        self._msaa_samples = self._output_samples or samples
        return self._msaa_samples

    @property
    def pass_count(self) -> int:
        return len(self._passes)

    @property
    def texture_count(self) -> int:
        return len(self._textures)

    @property
    def buffer_count(self) -> int:
        return len(self._buffers)

    @property
    def topology_sequence(self) -> List[Tuple[str, str]]:
        """Auto-recorded topology using stable IDs for user effect stages."""
        return list(self._topology)

    @property
    def injection_points(self) -> list:
        """All injection points declared via ``injection_point()``."""
        return list(self._injection_points_list)

    @property
    def effect_stages(self) -> list:
        """Pipeline-declared user attachment stages in topology order."""
        return list(self._effect_stages_list)

    def set_geometry_buffer_requirements(self, requirements) -> None:
        """Set semantic geometry buffers demanded by this graph revision."""
        normalized = {str(value or "").strip().lower() for value in requirements}
        normalized.discard("")
        self._geometry_buffer_requirements = frozenset(normalized)

    def require_geometry_buffers(self, requirements) -> None:
        self.set_geometry_buffer_requirements(
            set(self._geometry_buffer_requirements) | set(requirements)
        )

    @property
    def geometry_buffer_requirements(self):
        return self._geometry_buffer_requirements

    def needs_geometry_buffer(self, semantic: str) -> bool:
        return str(semantic or "").strip().lower() in self._geometry_buffer_requirements

    def publish_pass_result(self, source: str, buffers, *, materialize=None):
        """Publish the named buffers available after one stage or pass."""
        from Infernux.renderstack.pass_result import PassResult

        source_name = str(source or "").strip()
        if not source_name:
            raise ValueError("pass result source cannot be empty")
        if source_name in self._pass_results:
            raise ValueError(
                f"pass result source {source_name!r} is already published"
            )
        for semantic, resource in dict(buffers).items():
            if resource is not None and not isinstance(resource, (TextureHandle, BufferHandle)):
                raise TypeError(
                    f"pass resource {semantic!r} must be a graph TextureHandle or BufferHandle"
                )
            if resource is not None and not (
                isinstance(resource, TextureHandle) and self._owns_texture(resource)
                or isinstance(resource, BufferHandle) and self._owns_buffer(resource)
            ):
                raise ValueError(
                    f"pass resource {semantic!r} does not belong to this RenderGraph"
                )
        self._pass_result_revision += 1
        result = PassResult(
            source=source_name,
            revision=self._pass_result_revision,
            buffers=buffers,
            _materialize=materialize,
            _owner_graph=self,
        )
        self._pass_results[source_name] = result
        return result

    def derive_pass_result(self, source: str, parent, overrides):
        """Create a result after a pass writes one or more named buffers."""
        from Infernux.renderstack.pass_result import PassResult

        if not isinstance(parent, PassResult):
            raise TypeError("derive_pass_result() requires a PassResult parent")
        if parent._owner_graph is not self:
            raise ValueError("PassResult does not belong to this RenderGraph")
        merged = dict(parent.snapshot)
        merged.update(dict(overrides))
        return self.publish_pass_result(
            source,
            merged,
            materialize=parent._materialize,
        )

    def write_buffer(self, source: str, parent, name: str, texture):
        """Derive one result revision after writing a named virtual texture."""
        return self.derive_pass_result(source, parent, {name: texture})

    @property
    def pass_results(self):
        return dict(self._pass_results)

    @property
    def latest_pass_result(self):
        """Return the most recently published source-scoped result."""
        if not self._pass_results:
            return None
        return next(reversed(self._pass_results.values()))

    def get_pass_result(self, source: str):
        return self._pass_results.get(str(source or "").strip())

    # Transitional aliases are intentionally absent. This is a breaking API
    # revision: source-scoped PassResult replaces global geometry/scene maps.

    # ---- Resource creation ----

    @contextmanager
    def name_scope(self, prefix: str):
        """Namespace generated pass and transient-resource names.

        Reusable graph fragments may use readable local names while multiple
        instances coexist in one compiled graph.
        """
        normalized = str(prefix or "").strip().strip("/")
        if not normalized:
            raise ValueError("render graph name scope cannot be empty")
        parent = self._name_scopes[-1] if self._name_scopes else ""
        self._name_scopes.append(f"{parent}/{normalized}" if parent else normalized)
        try:
            yield self
        finally:
            self._name_scopes.pop()

    @contextmanager
    def effect_resources(self, resources: Mapping[str, TextureHandle]):
        """Bind semantic resources for EffectStages declared in this scope.

        Queue routes and layers own isolated graph textures whose native names
        are intentionally private.  Effects still consume the stable semantic
        names ``color`` and ``depth``.  This context maps those names without
        mutating graph-global aliases or leaking one route into the next.
        """
        if not isinstance(resources, Mapping):
            raise TypeError("effect resources must be a mapping")
        normalized: Dict[str, TextureHandle] = {}
        for name, handle in resources.items():
            resource_name = str(name or "").strip()
            if not resource_name:
                raise ValueError("effect resource names cannot be empty")
            if not isinstance(handle, TextureHandle):
                raise TypeError(
                    f"effect resource {resource_name!r} must be a TextureHandle"
                )
            if not self._owns_texture(handle):
                raise ValueError(
                    f"effect resource {resource_name!r} does not belong to this RenderGraph"
                )
            normalized[resource_name] = handle
        self._effect_resource_scopes.append(normalized)
        try:
            yield self
        finally:
            self._effect_resource_scopes.pop()

    @property
    def current_effect_resources(self) -> Mapping[str, TextureHandle]:
        """Semantic resource mapping active at the current topology point."""
        if not self._effect_resource_scopes:
            return {}
        return dict(self._effect_resource_scopes[-1])

    @contextmanager
    def pass_result(self, result):
        """Make one source-scoped result current while declaring stages.

        Effect compilation uses this explicit source instead of resolving
        process-global color/depth aliases. The callback may replace the top
        result with a derived revision after writing named buffers.
        """
        from Infernux.renderstack.pass_result import PassResult

        if not isinstance(result, PassResult):
            raise TypeError("pass_result() requires a PassResult")
        if result._owner_graph is not self:
            raise ValueError("PassResult does not belong to this RenderGraph")
        self._pass_result_scopes.append(result)
        try:
            yield self
        finally:
            self._pass_result_scopes.pop()

    @property
    def current_pass_result(self):
        return self._pass_result_scopes[-1] if self._pass_result_scopes else None

    def replace_current_pass_result(self, result) -> None:
        from Infernux.renderstack.pass_result import PassResult

        if not self._pass_result_scopes:
            raise RuntimeError("no active pass-result scope")
        if not isinstance(result, PassResult):
            raise TypeError("current pass result must be a PassResult")
        if result._owner_graph is not self:
            raise ValueError("PassResult does not belong to this RenderGraph")
        self._pass_result_scopes[-1] = result

    def resolve_effect_route_policy(self, stages):
        """Resolve mounted route effects without coupling the graph to a scene."""
        from Infernux.renderstack.route_policy import RoutePolicy

        stage_ids = tuple(getattr(stage, "stable_id", stage) for stage in stages)
        if not stage_ids:
            return RoutePolicy.INLINE
        if self._effect_route_policy_resolver is None:
            return RoutePolicy.ISOLATE_AND_COMPOSITE
        return RoutePolicy(self._effect_route_policy_resolver(stage_ids))

    def is_effect_stage_active(self, stage) -> bool:
        """Return whether a declared mount point has any enabled slot.

        Activity is intentionally separate from route ownership policy. A
        composite EffectGroup may legally contain additive and replacement
        effects even though that combination is ambiguous for an isolated
        render-queue route.
        """
        stable_id = getattr(stage, "stable_id", stage)
        if self._effect_stage_active_resolver is not None:
            return bool(self._effect_stage_active_resolver(stable_id))
        try:
            from Infernux.renderstack.route_policy import RoutePolicy

            return self.resolve_effect_route_policy((stage,)) is not RoutePolicy.INLINE
        except ValueError:
            return True

    def _scoped_name(self, name: str) -> str:
        raw = str(name)
        return f"{self._name_scopes[-1]}/{raw}" if self._name_scopes else raw

    def _find_texture_exact(self, name: str) -> Optional[TextureHandle]:
        return next((texture for texture in self._textures if texture.name == name), None)

    def _owns_texture(self, handle: TextureHandle) -> bool:
        return any(texture is handle for texture in self._textures)

    def _find_buffer_exact(self, name: str) -> Optional[BufferHandle]:
        return next((buffer for buffer in self._buffers if buffer.name == name), None)

    def _owns_buffer(self, handle: BufferHandle) -> bool:
        return any(buffer is handle for buffer in self._buffers)

    def create_texture(
        self,
        name: str,
        *,
        format: Format = Format.RGBA8_UNORM,
        camera_target: bool = False,
        size: "Optional[Tuple[int, int]]" = None,
        size_divisor: int = 0,
        samples: "Optional[int]" = None,
    ) -> TextureHandle:
        """Create a texture resource.

        Unified method — use keyword args for special textures::

            graph.create_texture("color", camera_target=True)
            graph.create_texture("depth", format=Format.D32_SFLOAT)
            graph.create_texture("shadow_map", format=Format.D32_SFLOAT, size=(4096, 4096))
            graph.create_texture("bloom_half", size_divisor=2)  # half-res

        Args:
            name: Unique string alias (e.g. ``"color"``, ``"depth"``).
            format: Texture format.
            camera_target: If ``True``, this is the camera's main color
                output. Resolution and format are determined by the engine.
            size: (width, height) custom resolution. ``None`` uses the
                scene render target size. Useful for shadow maps.
            size_divisor: Divide scene resolution by this value (>1).
                Mutually exclusive with *size*.
            samples: Texture sample count. ``None`` inherits frame MSAA for
                the camera target and scene-sized depth; other textures default
                to one sample. Explicit values are 0 (inherit), 1, 2, 4, or 8.
        """
        if size is not None and size_divisor > 0:
            raise ValueError(
                f"Texture '{name}' cannot use both size and size_divisor"
            )
        if size is not None:
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(
                    f"Texture '{name}' size must be positive, got {size}"
                )
        if size_divisor == 1:
            raise ValueError(
                f"Texture '{name}' size_divisor=1 has no effect; use 0 or >1"
            )
        if size_divisor < 0:
            raise ValueError(
                f"Texture '{name}' size_divisor must be >= 0"
            )
        if camera_target and format.is_depth:
            raise ValueError(
                f"Texture '{name}' cannot be a camera_target depth texture"
            )
        if samples is None:
            samples = (
                0
                if camera_target or (format.is_depth and size is None and size_divisor == 0)
                else 1
            )
        samples = int(samples)
        if samples not in (0, 1, 2, 4, 8):
            raise ValueError(
                f"Texture '{name}' samples must be 0, 1, 2, 4, or 8"
            )
        if camera_target and samples not in (0,):
            raise ValueError(
                f"Texture '{name}' is a camera target and must inherit frame MSAA"
            )

        resource_name = self._scoped_name(name)
        if (self._find_texture_exact(resource_name) is not None
                or self._find_buffer_exact(resource_name) is not None):
            raise ValueError(
                f"Resource '{resource_name}' already exists in graph '{self._name}'"
            )

        handle = TextureHandle(resource_name, format, is_camera_target=camera_target,
                               size=size, size_divisor=size_divisor, samples=samples)
        self._textures.append(handle)
        return handle

    def import_texture(self, name: str, texture, *, attachment: str = "color") -> TextureHandle:
        """Import a persistent RenderTexture attachment or a sampled Texture asset.

        A Texture asset is imported by GUID and is sample-only. Its sampler and
        immutable GPU publication come from the asset import pipeline, including
        Texture3D assets. ``attachment`` only applies to RenderTexture.

        For RenderTexture, ``color`` is the raster attachment; with MSAA, write it together with
        ``resolve`` using ``write_resolve`` and sample the resolve handle.
        ``depth`` uses the target's depth format and raster sample count.
        At one sample, color and resolve are the same graph identity.
        The first producer must initialize the attachment each execution.
        Input-only imports depend on this frame's active Camera/graph producer;
        the renderer orders views by resource dependencies, not old pixels.
        Missing producers and feedback cycles are rejected. Historical reads
        remain a separate, explicitly declared temporal-resource contract.
        Resize is observed without re-authoring the pipeline.
        """
        from Infernux.core.render_texture import RenderTexture
        from Infernux.lib import InxTexture

        resource_name = self._scoped_name(name)
        if isinstance(texture, InxTexture):
            if attachment != "color":
                raise ValueError("Texture assets are sample-only and do not expose attachments")
            guid = str(texture.guid).strip()
            if not guid:
                raise ValueError("Cannot import a Texture asset without a GUID")
            dimension = str(texture.dimension).lower()
            if dimension not in {"2d", "3d"}:
                raise ValueError(f"Unsupported Texture asset dimension '{texture.dimension}'")
            width = int(texture.pixel_width)
            height = int(texture.pixel_height)
            depth = int(texture.pixel_depth)
            if width <= 0 or height <= 0 or depth <= 0:
                raise ValueError("Cannot import a Texture asset with empty dimensions")
            occupied = self._find_texture_exact(resource_name) or self._find_buffer_exact(resource_name)
            if occupied is not None and (
                getattr(occupied, "asset_guid", "") != guid
                or bool(getattr(occupied, "is_volume", False)) != (dimension == "3d")
            ):
                raise ValueError(f"Resource '{resource_name}' already exists in graph '{self._name}'")
            if occupied is not None:
                occupied.size = (width, height)
                occupied.depth = depth
                return occupied
            handle = TextureHandle(resource_name, Format.UNDEFINED, size=(width, height), samples=1,
                                   temporal_role=GraphTextureRole.ASSET)
            handle.asset_guid = guid
            handle.depth = depth
            handle.is_volume = dimension == "3d"
            self._textures.append(handle)
            return handle

        if not isinstance(texture, RenderTexture):
            raise TypeError("import_texture requires an Infernux RenderTexture or Texture asset")
        if not texture.is_valid:
            raise ValueError("Cannot import a RenderTexture whose device has been destroyed")
        attachments = {"color": GraphTextureAttachment.COLOR, "depth": GraphTextureAttachment.DEPTH,
                       "resolve": GraphTextureAttachment.RESOLVE}
        if attachment not in attachments:
            raise ValueError("RenderTexture attachment must be 'color', 'depth', or 'resolve'")
        if attachment == "depth" and texture.depth_format == PixelFormat.UNDEFINED:
            raise ValueError("RenderTexture has no depth attachment")
        if attachment == "resolve" and texture.samples == 1:
            attachment = "color"
        selected = attachments[attachment]
        occupied = self._find_texture_exact(resource_name) or self._find_buffer_exact(resource_name)
        if occupied is not None and (getattr(occupied, "render_texture", None) is not texture or
                                     occupied.attachment != selected):
            raise ValueError(f"Resource '{resource_name}' already exists in graph '{self._name}'")
        for existing in self._textures:
            if existing.render_texture is texture and existing.attachment == selected:
                return existing
        format = texture.depth_format if attachment == "depth" else texture.format
        samples = 1 if attachment == "resolve" else texture.samples
        handle = TextureHandle(resource_name, format, size=(texture.width, texture.height), samples=samples,
                               temporal_role=GraphTextureRole.PERSISTENT)
        handle.render_texture = texture
        handle.attachment = selected
        self._textures.append(handle)
        return handle

    def create_temporal_history(
        self,
        name: str,
        *,
        format: Format = Format.RGBA16_SFLOAT,
        size: Optional[Tuple[int, int]] = None,
        size_divisor: int = 0,
    ) -> Tuple[TextureHandle, TextureHandle]:
        """Create a per-view, single-sample history read/write pair.

        The native renderer owns and ping-pongs both images. They survive
        ordinary graph execution but are invalidated by view/target changes.
        The first read after invalidation is zero. Write the output every
        execution; raster and copy writes are supported. Camera jitter is a
        separate explicit request via set_temporal_jitter().
        """
        if format.is_depth:
            raise ValueError("temporal history must use a color format")
        local_name = str(name or "").strip()
        if not local_name:
            raise ValueError("temporal history name cannot be empty")
        base = self._scoped_name(local_name)
        read_name = f"{base}/read"
        write_name = f"{base}/write"
        if any(self._find_texture_exact(n) is not None or self._find_buffer_exact(n) is not None
               for n in (read_name, write_name)):
            raise ValueError(f"Temporal history '{base}' already exists")
        read = self.create_texture(f"{local_name}/read", format=format, size=size,
                                   size_divisor=size_divisor, samples=1)
        write = self.create_texture(f"{local_name}/write", format=format, size=size,
                                    size_divisor=size_divisor, samples=1)
        read.temporal_role = GraphTextureRole.TEMPORAL_READ
        write.temporal_role = GraphTextureRole.TEMPORAL_WRITE
        read.temporal_key = write.temporal_key = base
        return read, write

    def get_texture(self, name: str) -> Optional[TextureHandle]:
        """Look up a texture by its string alias.

        Returns:
            ``TextureHandle`` or ``None`` if not found.
        """
        exact = self._find_texture_exact(str(name))
        if exact is not None or not self._name_scopes:
            return exact
        return self._find_texture_exact(self._scoped_name(name))

    def create_buffer(
        self,
        name: str,
        byte_size: int,
        *,
        storage: bool = True,
        indirect: bool = False,
        transfer_source: bool = False,
        transfer_destination: bool = False,
    ) -> BufferHandle:
        """Create a transient graph buffer with backend-neutral usage flags."""
        if byte_size <= 0:
            raise ValueError(f"Buffer '{name}' byte_size must be positive")
        resource_name = self._scoped_name(name)
        if (self._find_texture_exact(resource_name) is not None
                or self._find_buffer_exact(resource_name) is not None):
            raise ValueError(f"Resource '{resource_name}' already exists in graph '{self._name}'")
        usage = int(GraphBufferUsage.NONE)
        if storage:
            usage |= int(GraphBufferUsage.STORAGE)
        if indirect:
            usage |= int(GraphBufferUsage.INDIRECT)
        if transfer_source:
            usage |= int(GraphBufferUsage.TRANSFER_SOURCE)
        if transfer_destination:
            usage |= int(GraphBufferUsage.TRANSFER_DESTINATION)
        if usage == int(GraphBufferUsage.NONE):
            raise ValueError(f"Buffer '{name}' must declare at least one usage")
        handle = BufferHandle(resource_name, int(byte_size), usage)
        self._buffers.append(handle)
        return handle

    def import_buffer(self, name: str, buffer) -> BufferHandle:
        """Import an engine-owned uint32 GPU buffer into this view graph.

        The caller retains the ``inx.buffer`` object. The graph retains its
        native allocation until its last submitted frame has retired.
        """
        resource_name = self._scoped_name(name)
        if (self._find_texture_exact(resource_name) is not None
                or self._find_buffer_exact(resource_name) is not None):
            raise ValueError(f"Resource '{resource_name}' already exists in graph '{self._name}'")
        if (getattr(buffer, "device", None) != "gpu"
                or getattr(buffer, "dtype", None) != "uint32"
                or getattr(buffer, "closed", True)
                or getattr(buffer, "_native", None) is None):
            raise ValueError("import_buffer requires an open uint32 GPU inx.buffer")
        if buffer._byte_offset != 0 or buffer.nbytes != buffer._native.byte_size:
            raise ValueError("import_buffer requires the complete GPU allocation, not a buffer view")
        handle = BufferHandle(resource_name, buffer.nbytes, int(GraphBufferUsage.STORAGE))
        handle.compute_buffer = buffer._native
        self._buffers.append(handle)
        return handle

    def get_buffer(self, name: str) -> Optional[BufferHandle]:
        """Look up a graph buffer by alias."""
        exact = self._find_buffer_exact(str(name))
        if exact is not None or not self._name_scopes:
            return exact
        return self._find_buffer_exact(self._scoped_name(name))

    # ---- Query helpers ----

    def has_pass(self, name: str) -> bool:
        """Check if a pass with *name* has already been added."""
        raw = str(name)
        scoped = self._scoped_name(raw)
        return any(p._name == raw or p._name == scoped for p in self._passes)

    def has_injection_point(self, name: str) -> bool:
        """Check if an injection point with *name* has been declared."""
        return any(ip.name == name for ip in self._injection_points_list)

    def has_effect_stage(self, stable_id: str) -> bool:
        """Return whether an exact stage ID is declared."""
        return any(stage.stable_id == stable_id for stage in self._effect_stages_list)

    # ---- Injection points ----

    def injection_point(
        self,
        name: str,
        *,
        display_name: str = "",
        resources: Optional[set] = None,
    ) -> None:
        """Declare an injection point at the current topology position.

        Pipeline authors use injection points to name stable topology boundaries.
        User-facing effects bind only to explicit EffectStages.

        Args:
            name: Unique identifier (e.g. ``"after_opaque"``).
            display_name: Editor label (auto-generated from *name* if empty).
            resources: Guaranteed-available resource names at this point.
        """
        from Infernux.renderstack.injection_point import InjectionPoint

        ip = InjectionPoint(
            name=name,
            display_name=display_name or name.replace("_", " ").title(),
            resource_state=resources if resources is not None else {"color", "depth"},
        )
        self._injection_points_list.append(ip)
        self._topology.append(("ip", ip.display_name))

        if self._injection_callback is not None:
            self._injection_callback(name)

    # ---- Pipeline-declared user EffectStages ----

    def effect_stage(
        self,
        stable_id: str,
        *,
        scope="composite",
        display_name: str = "",
        inputs=None,
        outputs=None,
        capabilities=None,
    ):
        """Declare one stable user-facing RenderEffect attachment stage.

        The stage is topology, not scene data. A RenderStack may bind ordered
        slots to this declaration but cannot invent additional stages.
        """
        from Infernux.renderstack.effect_stage import (
            EffectResourceContract,
            EffectStage,
        )

        stage = EffectStage(
            stable_id=stable_id,
            scope=scope,
            display_name=display_name,
            contract=EffectResourceContract(
                inputs=frozenset(inputs or ()),
                outputs=frozenset(outputs or ()),
                capabilities=frozenset(capabilities or ()),
            ),
        )
        if self.has_effect_stage(stage.stable_id):
            raise ValueError(
                f"EffectStage IDs must be unique in graph '{self._name}': "
                f"{stage.stable_id!r}"
            )

        self._effect_stages_list.append(stage)
        self._topology.append(("effect_stage", stage.stable_id))
        if self._effect_stage_callback is not None:
            self._effect_stage_callback(stage)
        return stage

    def effects(self, stable_id: str, **kwargs):
        """Pipeline-author shorthand for :meth:`effect_stage`."""
        return self.effect_stage(stable_id, **kwargs)

    # ---- Convenience: Camera UI + post-process + Screen UI sections ----

    def camera_ui_section(self, *, resources: "set | None" = None,
                          world_ui_layer_mask: int = 0xffffffff) -> None:
        """Draw Camera UI and expose the composite immediately afterwards."""
        res = resources or {"color"}
        if not self.has_pass("_WorldUI"):
            with self.add_pass("_WorldUI") as p:
                p.write_color("color")
                p.write_depth("depth")
                p.draw_world_ui(layer_mask=world_ui_layer_mask)
        if not self.has_pass("_ScreenUI_Camera"):
            with self.add_pass("_ScreenUI_Camera") as p:
                p.write_color("color")
                p.draw_screen_ui(list="camera")

        if not self.has_effect_stage("after_camera_ui"):
            self.effects(
                "after_camera_ui",
                scope="composite",
                display_name="After Camera UI",
                inputs=res,
                outputs={"color"},
                capabilities={"fullscreen"},
            )

    def screen_ui_overlay_section(self, *, resources: "set | None" = None) -> None:
        """Encode the scene for display, draw Screen UI, then expose it."""
        res = resources or {"color"}
        self.display_encode_section()

        if not self.has_pass("_ScreenUI_Overlay"):
            with self.add_pass("_ScreenUI_Overlay") as p:
                p.write_color("color")
                p.draw_screen_ui(list="overlay")

        if not self.has_effect_stage("after_screen_ui"):
            self.effects(
                "after_screen_ui",
                scope="composite",
                display_name="After Screen UI",
                inputs=res,
                outputs={"color"},
                capabilities={"fullscreen", "display_space"},
            )

    def screen_ui_section(self, *, resources: "set | None" = None,
                          world_ui_layer_mask: int = 0xffffffff) -> None:
        """Insert the canonical Camera UI, post-process and Screen UI tail.

        This is a convenience shortcut that emits::

            _WorldUI                  (draw_world_ui, layer-filtered)
            _ScreenUI_Camera          (draw_screen_ui list="camera")
            after_camera_ui           (effect stage)
            before_post_process       (legacy injection point)
            final                     (effect stage)
            after_post_process        (legacy injection point)
            _DisplayEncode            (linear to display encoding)
            _ScreenUI_Overlay         (draw_screen_ui list="overlay")
            after_screen_ui           (effect stage, display space)

        Custom pipelines can call this at the desired topology position. The
        low-level helper is explicit, while RenderStack finalization appends
        the display-space Screen UI tail when a custom pipeline omits it.

        Override behavior: each element is only inserted when missing, so
        users may pre-declare one or more reserved names and let this method
        fill the rest without duplication.

        Args:
            resources: Resource set advertised to injection points.
                       Defaults to ``{"color"}``.
            world_ui_layer_mask: GameObject layers drawn by the inserted World
                                 UI pass. Defaults to all layers. Predeclared
                                 passes retain their own masks.
        """
        res = resources or {"color"}

        self.camera_ui_section(resources=res, world_ui_layer_mask=world_ui_layer_mask)

        if not self.has_injection_point("before_post_process"):
            self.injection_point("before_post_process", resources=res)
        if not self.has_effect_stage("final"):
            self.effects(
                "final",
                scope="composite",
                display_name="Final Post Processing",
                inputs=res,
                outputs={"color"},
                capabilities={"fullscreen", "hdr_to_display"},
            )
        if not self.has_injection_point("after_post_process"):
            self.injection_point("after_post_process", resources=res)

        self.screen_ui_overlay_section(resources=res)

    def display_encode_section(self) -> None:
        """Insert the built-in linear → sRGB display-encode passes.

        The swapchain and the editor viewport are UNORM surfaces without
        hardware sRGB encoding, so every pipeline must end with an explicit
        display encode. Scene rendering and all post-process effects operate
        in linear HDR; this is the single place where gamma is applied.

        Idempotent: calling it twice (e.g. from ``screen_ui_section`` and the
        RenderStack build safety net) inserts the passes only once.
        """
        if self.has_pass("_DisplayEncode"):
            return

        self._linear_output = self.get_texture("color").name
        self.create_texture("_display_encode", format=Format.RGBA16_SFLOAT)
        with self.add_pass("_DisplayEncode") as p:
            p.set_texture("_SourceTex", "color")
            p.write_color("_display_encode")
            # The native per-camera render context overwrites these values at
            # draw time without rebuilding the graph.
            p.set_param("dithering", 0.0)
            p.set_param("stopNaNs", 0.0)
            p.fullscreen_quad("Display Encode")
        with self.add_pass("_DisplayEncode_Commit") as p:
            p.set_texture("_SourceTex", "_display_encode")
            p.write_color("color")
            p.fullscreen_quad("Fullscreen Blit")

    # ---- Pass management ----

    def add_pass(self, name: str) -> RenderPassBuilder:
        """Add a render pass to the graph.

        Returns a ``RenderPassBuilder`` (also a context manager) that you
        use to configure the pass::

            with graph.add_pass("OpaquePass") as p:
                p.write_color("color")
                p.write_depth("depth")
                p.draw_renderers(queue_range=(0, 2500))

        The pass is appended to the topology sequence automatically.
        """
        pass_name = self._scoped_name(name)
        builder = RenderPassBuilder(pass_name, graph=self, pass_type="raster")
        self._passes.append(builder)
        self._topology.append(("pass", pass_name))
        return builder

    def _add_typed_pass(self, name: str, pass_type: str) -> RenderPassBuilder:
        pass_name = self._scoped_name(name)
        builder = RenderPassBuilder(pass_name, graph=self, pass_type=pass_type)
        self._passes.append(builder)
        self._topology.append(("pass", pass_name))
        return builder

    def add_copy_pass(self, name: str) -> RenderPassBuilder:
        """Add a transfer-domain texture or buffer copy pass."""
        return self._add_typed_pass(name, "copy")

    def add_present_pass(self, name: str) -> RenderPassBuilder:
        """Add the final graph export pass."""
        return self._add_typed_pass(name, "present")

    def remove_pass(self, name: str) -> "RenderPassBuilder | None":
        """Remove a pass by name and return it, or ``None`` if not found.

        Also removes the corresponding topology entry.
        """
        removed = None
        for i, p in enumerate(self._passes):
            if p._name == name:
                removed = self._passes.pop(i)
                break
        if removed is not None:
            for i, (kind, label) in enumerate(self._topology):
                if kind == "pass" and label == name:
                    self._topology.pop(i)
                    break
        return removed

    def append_pass(self, builder: "RenderPassBuilder") -> None:
        """Re-append a previously removed pass at the end of the topology."""
        if not isinstance(builder, RenderPassBuilder) or builder._graph is not self:
            raise ValueError("RenderPassBuilder does not belong to this RenderGraph")
        self._passes.append(builder)
        self._topology.append(("pass", builder._name))

    # ---- Output ----

    def set_output(self, texture) -> None:
        """Mark a texture as the final graph output.

        Args:
            texture: Texture alias (``str``) or ``TextureHandle``.
        """
        if isinstance(texture, str):
            handle = self.get_texture(texture)
            if handle is None:
                raise ValueError(
                    f"set_output: Texture '{texture}' not found in graph "
                    f"'{self._name}'"
                )
            self._output = handle.name
        else:
            if not isinstance(texture, TextureHandle) or not self._owns_texture(texture):
                raise ValueError("set_output: TextureHandle does not belong to this RenderGraph")
            self._output = texture.name

    # ---- Validation & finalization ----

    def validate_no_ip_before_first_pass(self) -> None:
        """Reject user extension points that precede the first render pass.

        The new API forbids IPs before the very first pass (use
        ``after_opaque`` etc. instead).
        """
        for kind, _label in self._topology:
            if kind == "pass":
                return  # first entry is a pass — OK
            if kind in {"ip", "effect_stage"}:
                raise ValueError(
                    f"Graph '{self._name}': {kind} declared before the first "
                    "pass. This user extension point requires an upstream result."
                )

    def _validate_graph(self) -> None:
        self.validate_no_ip_before_first_pass()

        texture_map = {tex.name: tex for tex in self._textures}
        buffer_map = {buffer.name: buffer for buffer in self._buffers}
        pass_names = set()

        # Warn if multiple textures claim camera_target — they alias to the
        # same physical swapchain image, which is almost never intended.
        camera_targets = [t.name for t in self._textures if t.is_camera_target]
        if len(camera_targets) > 1:
            warnings.warn(
                f"[RenderGraph '{self._name}'] Multiple camera_target textures "
                f"({', '.join(camera_targets)}). All camera_target textures "
                f"alias to the same swapchain image — only one should be "
                f"camera_target=True.",
                stacklevel=3,
            )

        for tex in self._textures:
            if tex.size is not None and tex.size_divisor > 0:
                raise ValueError(
                    f"Texture '{tex.name}' cannot use both size and size_divisor"
                )
            if tex.size_divisor == 1:
                raise ValueError(
                    f"Texture '{tex.name}' size_divisor=1 has no effect; use 0 or >1"
                )

        produced = set()
        for p in self._passes:
            if p._name in pass_names:
                raise ValueError(
                    f"Graph '{self._name}' contains duplicate pass name '{p._name}'"
                )
            pass_names.add(p._name)
            self._validate_pass(p, texture_map, buffer_map)
            if p._action == "fullscreen_quad":
                loads = []
                if (p._alpha_blend or p._depth_test is not None) and p._clear_color is None:
                    loads.extend(p._write_colors.values())
                if p._write_depth is not None and p._clear_depth is None:
                    loads.append(p._write_depth)
                for name in loads:
                    if name not in produced and texture_map[name].temporal_role != GraphTextureRole.PERSISTENT:
                        raise ValueError(f"Fullscreen pass '{p._name}' must clear or load an earlier output: {name}")
            produced.update(p._write_colors.values())
            if p._write_depth is not None:
                produced.add(p._write_depth)
            if p._resolve_color is not None:
                produced.add(p._resolve_color)
            if p._action == "copy_texture":
                produced.add(p._destination_resource)

        if self._output is not None and self._output not in texture_map:
            raise ValueError(
                f"Graph '{self._name}' output '{self._output}' does not exist"
            )
        if self._output is not None and texture_map[self._output].temporal_role == GraphTextureRole.ASSET:
            raise ValueError("A sampled Texture asset cannot be the graph output")

    def _validate_pass(
        self,
        p: RenderPassBuilder,
        texture_map: Dict[str, TextureHandle],
        buffer_map: Dict[str, BufferHandle],
    ) -> None:
        raster_actions = {
            "none", "draw_renderers", "draw_skybox", "custom",
            "draw_shadow_casters", "draw_world_ui", "draw_screen_ui", "fullscreen_quad",
        }
        allowed_actions = {
            "raster": raster_actions,
            "copy": {"copy_texture", "copy_buffer"},
            "present": {"present"},
        }
        if p._pass_type not in allowed_actions or p._action not in allowed_actions[p._pass_type]:
            raise ValueError(
                f"Pass '{p._name}' action '{p._action}' is invalid for {p._pass_type} pass"
            )
        if p._pass_type != "raster" and (
            p._write_colors or p._write_depth is not None
            or p._resolve_color is not None
            or p._clear_color is not None or p._clear_depth is not None
            or p._reads
        ):
            raise ValueError(
                f"Pass '{p._name}' uses raster texture attachments in a {p._pass_type} pass"
            )
        if p._action == "draw_shadow_casters" and p._write_colors:
            raise ValueError(
                f"Pass '{p._name}' is depth-only and cannot write color targets"
            )

        if p._clear_depth is not None and p._write_depth is None:
            raise ValueError(
                f"Pass '{p._name}' clears depth but has no depth output"
            )

        if p._action == "fullscreen_quad":
            if sorted(p._write_colors) != [0]:
                raise ValueError(f"Fullscreen pass '{p._name}' requires one color output at slot 0")
            if p._depth_test is not None and p._write_depth is None:
                raise ValueError(f"Fullscreen pass '{p._name}' requires a depth output for depth_test")
            if p._write_depth is not None and (p._write_depth in p._reads or p._write_depth in p._input_bindings.values()):
                raise ValueError(f"Fullscreen pass '{p._name}' cannot sample its attached depth; copy it first")
        elif p._depth_test is not None or p._depth_write or p._alpha_blend:
            raise ValueError(f"Pass '{p._name}' uses fullscreen state with another draw action")

        if p._action == "draw_renderers":
            slots = sorted(p._write_colors)
            if slots != list(range(len(slots))):
                raise ValueError(
                    f"Pass '{p._name}' color slots must be contiguous from zero"
                )
            if p._material_pass == "depth":
                if p._write_colors or p._write_depth is None:
                    raise ValueError(
                        f"Depth pass '{p._name}' requires one depth output and no color outputs"
                    )
            elif p._material_pass == "picking":
                colors = [texture_map[name] for _, name in sorted(p._write_colors.items())]
                if (
                    len(colors) != 1
                    or colors[0].format != Format.RG32_UINT
                    or p._write_depth is None
                ):
                    raise ValueError(
                        f"Picking pass '{p._name}' requires RG32_UINT color[0] and a depth output"
                    )
            elif p._material_pass == "motion":
                colors = [texture_map[name] for _, name in sorted(p._write_colors.items())]
                depth_reads = [texture_map[name] for name in p._reads if texture_map[name].is_depth]
                if (
                    len(colors) != 1
                    or colors[0].format != Format.RG16_SFLOAT
                    or len(depth_reads) != 1
                    or p._write_depth is not None
                ):
                    raise ValueError(
                        f"Motion pass '{p._name}' requires one RG16_SFLOAT color output, "
                        "one readable depth texture, and no depth write"
                    )
            elif p._material_pass == "normal":
                colors = [texture_map[name] for _, name in sorted(p._write_colors.items())]
                depth_reads = [texture_map[name] for name in p._reads if texture_map[name].is_depth]
                if (
                    len(colors) != 1
                    or colors[0].format != Format.RGBA16_SFLOAT
                    or len(depth_reads) != 1
                    or p._write_depth is not None
                ):
                    raise ValueError(
                        f"Normal pass '{p._name}' requires one RGBA16_SFLOAT color output, "
                        "one readable depth texture, and no depth write"
                    )
            elif not p._write_colors:
                raise ValueError(
                    f"Material pass '{p._name}' requires at least one color output"
                )

        for buffer_name, _access in p._buffer_accesses:
            buffer = buffer_map.get(buffer_name)
            if buffer is None:
                raise ValueError(
                    f"Pass '{p._name}' references unknown buffer '{buffer_name}'"
                )
            required_usage = {
                "storage_read": int(GraphBufferUsage.STORAGE),
                "storage_write": int(GraphBufferUsage.STORAGE),
                "indirect_read": int(GraphBufferUsage.INDIRECT),
                "transfer_read": int(GraphBufferUsage.TRANSFER_SOURCE),
                "transfer_write": int(GraphBufferUsage.TRANSFER_DESTINATION),
            }[_access]
            if not buffer.usage & required_usage:
                raise ValueError(
                    f"Pass '{p._name}' buffer '{buffer_name}' does not declare "
                    f"the usage required by '{_access}'"
                )

        if p._action == "copy_texture":
            if p._source_resource not in texture_map or p._destination_resource not in texture_map:
                raise ValueError(f"Copy pass '{p._name}' references an unknown texture")
            if p._source_resource == p._destination_resource:
                raise ValueError(f"Copy pass '{p._name}' requires distinct textures")
            source = texture_map[p._source_resource]
            destination = texture_map[p._destination_resource]
            if (source.temporal_role == GraphTextureRole.ASSET
                    or destination.temporal_role == GraphTextureRole.ASSET):
                raise ValueError(f"Copy pass '{p._name}' cannot use a sample-only Texture asset")
            if source.is_camera_target or destination.is_camera_target:
                raise ValueError(
                    f"Copy pass '{p._name}' requires transient textures; use present() "
                    "or a fullscreen pass for a camera target"
                )
            if source.format != destination.format:
                raise ValueError(f"Copy pass '{p._name}' requires matching texture formats")
        elif p._action == "copy_buffer":
            if p._source_resource not in buffer_map or p._destination_resource not in buffer_map:
                raise ValueError(f"Copy pass '{p._name}' references an unknown buffer")
            if p._source_resource == p._destination_resource:
                raise ValueError(f"Copy pass '{p._name}' requires distinct buffers")
            maximum = min(
                buffer_map[p._source_resource].byte_size,
                buffer_map[p._destination_resource].byte_size,
            )
            if p._copy_bytes > maximum:
                raise ValueError(f"Copy pass '{p._name}' exceeds the smaller buffer")
        elif p._action == "present" and p._source_resource not in texture_map:
            raise ValueError(f"Present pass '{p._name}' references an unknown texture")
        elif p._action == "present" and texture_map[p._source_resource].is_depth:
            raise ValueError(f"Present pass '{p._name}' cannot export a depth texture")

        for read_name in p._reads:
            if read_name not in texture_map:
                raise ValueError(
                    f"Pass '{p._name}' reads unknown texture '{read_name}'"
                )

        for slot, tex_name in p._write_colors.items():
            tex = texture_map.get(tex_name)
            if tex is None:
                raise ValueError(
                    f"Pass '{p._name}' writes unknown color target '{tex_name}'"
                )
            if tex.is_depth:
                raise ValueError(
                    f"Pass '{p._name}' writes depth texture '{tex_name}' as color[{slot}]"
                )
            if tex.temporal_role == GraphTextureRole.ASSET:
                raise ValueError(f"Pass '{p._name}' cannot write sample-only Texture asset '{tex_name}'")

        if p._write_depth is not None:
            tex = texture_map.get(p._write_depth)
            if tex is None:
                raise ValueError(
                    f"Pass '{p._name}' writes unknown depth target '{p._write_depth}'"
                )
            if not tex.is_depth:
                raise ValueError(
                    f"Pass '{p._name}' writes color texture '{p._write_depth}' as depth"
                )
            if tex.temporal_role == GraphTextureRole.ASSET:
                raise ValueError(f"Pass '{p._name}' cannot write sample-only Texture asset '{p._write_depth}'")

        attachment_names = list(p._write_colors.values())
        if p._write_depth is not None:
            attachment_names.append(p._write_depth)
        attachment_samples = {
            (
                self._msaa_samples
                if texture_map[name].samples == 0 and self._msaa_samples > 0
                else texture_map[name].samples
            )
            for name in attachment_names
        }
        known_attachment_samples = {value for value in attachment_samples if value > 0}
        if len(known_attachment_samples) > 1:
            raise ValueError(
                f"Pass '{p._name}' attachments use different sample counts: "
                f"{sorted(known_attachment_samples)}"
            )

        if p._resolve_color is not None:
            resolve = texture_map.get(p._resolve_color)
            if resolve is None:
                raise ValueError(
                    f"Pass '{p._name}' resolves into unknown texture '{p._resolve_color}'"
                )
            if resolve.is_depth or resolve.is_camera_target:
                raise ValueError(
                    f"Pass '{p._name}' resolve target must be a non-camera color texture"
                )
            if resolve.temporal_role == GraphTextureRole.ASSET:
                raise ValueError(f"Pass '{p._name}' cannot resolve into a sample-only Texture asset")
            if sorted(p._write_colors) != [0]:
                raise ValueError(
                    f"Pass '{p._name}' resolve requires exactly one color output at slot 0"
                )
            source = texture_map[p._write_colors[0]]
            source_samples = (
                self._msaa_samples
                if source.samples == 0 and self._msaa_samples > 0
                else source.samples
            )
            if source_samples not in (2, 4, 8):
                raise ValueError(
                    f"Pass '{p._name}' resolve source must be multisampled"
                )
            if resolve.samples != 1:
                raise ValueError(
                    f"Pass '{p._name}' resolve target must use samples=1"
                )
            if source.format != resolve.format:
                raise ValueError(
                    f"Pass '{p._name}' resolve source and target formats must match"
                )
            if source.size != resolve.size or source.size_divisor != resolve.size_divisor:
                raise ValueError(
                    f"Pass '{p._name}' resolve source and target extents must match"
                )

        for sampler_name, tex_name in p._input_bindings.items():
            if tex_name not in texture_map and tex_name not in buffer_map:
                raise ValueError(
                    f"Pass '{p._name}' input '{sampler_name}' references unknown resource '{tex_name}'"
                )
            if tex_name in buffer_map and p._action != "fullscreen_quad":
                raise ValueError(f"Pass '{p._name}' binds a buffer outside a fullscreen pass")

        if p._action == "draw_shadow_casters" and p._write_depth is None:
            raise ValueError(
                f"Pass '{p._name}' is a shadow caster pass and requires a depth output"
            )

    # ---- Build ----

    def build(self) -> "RenderGraphDescription":
        """Build the graph into a C++ RenderGraphDescription.

        Validates the graph topology and produces the POD structure that
        C++ expects. Raises ValueError if there are validation issues.

        ``before_post_process`` and ``after_post_process`` injection points
        are **always** auto-injected at the end of the topology if the
        pipeline did not already declare them (via ``screen_ui_section()``
        or explicit ``injection_point()`` calls).  This guarantees that
        user passes targeting those slots always have somewhere to attach,
        regardless of the pipeline implementation.

        Returns:
            RenderGraphDescription (C++ POD) ready for
            ``SceneRenderGraph.apply_python_graph()``.
        """
        if not self._passes:
            raise ValueError(f"Graph '{self._name}' has no passes")

        self._validate_graph()

        # Auto-inject before/after_post_process injection points when the
        # pipeline didn't declare them.  Uses has_injection_point() so
        # pipelines that already define these (e.g. via screen_ui_section)
        # are not affected.
        _auto_res = {"color"}
        if not self.has_injection_point("before_post_process"):
            self.injection_point("before_post_process", resources=_auto_res)
        if not self.has_injection_point("after_post_process"):
            self.injection_point("after_post_process", resources=_auto_res)

        if self._output is None:
            # Auto-set output to camera_target if exists
            for tex in self._textures:
                if tex.is_camera_target:
                    self._output = tex.name
                    break

        if self._output is None:
            raise ValueError(
                f"Graph '{self._name}' has no output. "
                "Call graph.set_output(texture)."
            )

        return self._build_native()

    def _build_native(self):
        """Build using native C++ types."""
        desc = RenderGraphDescription()
        desc.name = self._name
        desc.source_revision = next(_SOURCE_REVISION_COUNTER)

        # Build texture list — construct full list then assign (pybind11
        # vectors return copies, so append() on a property doesn't work).
        tex_list = []
        for tex in self._textures:
            td = GraphTextureDesc()
            td.name = tex.name
            td.format = tex.format
            td.is_backbuffer = tex.is_camera_target
            td.is_depth = tex.is_depth
            if tex.size is not None:
                td.width = tex.size[0]
                td.height = tex.size[1]
            if tex.size_divisor > 0:
                td.size_divisor = tex.size_divisor
            td.samples = tex.samples
            td.role = tex.temporal_role
            td.temporal_key = tex.temporal_key
            if tex.render_texture is not None:
                td.render_texture = tex.render_texture._native
                td.attachment = tex.attachment
                td.width, td.height = tex.render_texture.width, tex.render_texture.height
            if tex.asset_guid:
                td.asset_guid = tex.asset_guid
                td.depth = tex.depth
                td.is_volume = tex.is_volume
            tex_list.append(td)
        desc.textures = tex_list

        buffer_list = []
        for buffer in self._buffers:
            bd = GraphBufferDesc()
            bd.name = buffer.name
            bd.byte_size = buffer.byte_size
            bd.usage = buffer.usage
            bd.compute_buffer = buffer.compute_buffer
            buffer_list.append(bd)
        desc.buffers = buffer_list

        # Build pass list
        _command_map = {
            "draw_renderers": GraphCommandType.DRAW_RENDERERS,
            "draw_skybox": GraphCommandType.DRAW_SKYBOX,
            "draw_shadow_casters": GraphCommandType.DRAW_SHADOW_CASTERS,
            "draw_world_ui": GraphCommandType.DRAW_WORLD_UI,
            "draw_screen_ui": GraphCommandType.DRAW_SCREEN_UI,
            "fullscreen_quad": GraphCommandType.FULLSCREEN_QUAD,
            "copy_texture": GraphCommandType.COPY_TEXTURE,
            "copy_buffer": GraphCommandType.COPY_BUFFER,
            "present": GraphCommandType.PRESENT,
        }
        _pass_type_map = {
            "raster": GraphPassType.RASTER,
            "copy": GraphPassType.COPY,
            "present": GraphPassType.PRESENT,
        }
        _buffer_access_map = {
            "storage_read": GraphBufferAccessType.STORAGE_READ,
            "storage_write": GraphBufferAccessType.STORAGE_WRITE,
            "indirect_read": GraphBufferAccessType.INDIRECT_READ,
            "transfer_read": GraphBufferAccessType.TRANSFER_READ,
            "transfer_write": GraphBufferAccessType.TRANSFER_WRITE,
        }
        _material_pass_map = {
            "forward": MaterialPassType.FORWARD,
            "forward_plus": MaterialPassType.FORWARD_PLUS,
            "gbuffer": MaterialPassType.GBUFFER,
            "depth": MaterialPassType.DEPTH,
            "shadow": MaterialPassType.SHADOW,
            "picking": MaterialPassType.PICKING,
            "motion": MaterialPassType.MOTION,
            "normal": MaterialPassType.NORMAL,
            "base_color": MaterialPassType.BASE_COLOR,
        }
        _material_filter_map = {
            "all": GraphMaterialFilter.ALL,
            "deferred_compatible": GraphMaterialFilter.DEFERRED_COMPATIBLE,
            "deferred_unsupported": GraphMaterialFilter.DEFERRED_UNSUPPORTED,
        }

        pass_list = []
        for p in self._passes:
            pd = GraphPassDesc()
            pd.name = p._name
            pd.type = _pass_type_map[p._pass_type]
            pd.read_textures = list(p._reads)
            # MRT support: serialize write_colors as list of (slot, name) pairs
            pd.write_colors = list(p._write_colors.items())
            pd.write_depth = p._write_depth or ""
            pd.resolve_color = p._resolve_color or ""
            accesses = []
            for resource, access_type in p._buffer_accesses:
                access = GraphBufferAccessDesc()
                access.resource = resource
                access.type = _buffer_access_map[access_type]
                accesses.append(access)
            pd.buffer_accesses = accesses
            pd.side_effect = p._side_effect

            if p._clear_color is not None:
                pd.clear_color = True
                pd.clear_color_r = p._clear_color[0]
                pd.clear_color_g = p._clear_color[1]
                pd.clear_color_b = p._clear_color[2]
                pd.clear_color_a = p._clear_color[3]
            else:
                pd.clear_color = False

            if p._clear_depth is not None:
                pd.clear_depth = True
                pd.clear_depth_value = p._clear_depth
            else:
                pd.clear_depth = False

            command_type = _command_map.get(p._action)
            if command_type is not None:
                command = GraphCommandDesc()
                command.type = command_type
                command.material_pass = _material_pass_map[p._material_pass]
                command.material_filter = _material_filter_map[p._material_filter]
                command.queue_min = p._queue_min
                command.queue_max = p._queue_max
                command.sort_mode = p._sort_mode
                command.pass_tag = p._pass_tag
                command.override_material = p._override_material
                if p._renderer_selection is not None:
                    command.renderer_selection = p._renderer_selection._native
                command.input_bindings = list(p._input_bindings.items())
                command.light_index = p._light_index
                command.screen_ui_list = p._screen_ui_list
                if p._action == "draw_world_ui":
                    command.world_ui_layer_mask = p._world_ui_layer_mask
                command.shader_name = p._shader_name
                command.depth_test = p._depth_test is not None
                command.depth_write = p._depth_write
                command.depth_compare = p._depth_test if p._depth_test is not None else DepthCompare.ALWAYS
                command.alpha_blend = p._alpha_blend
                command.parameter_block = p._parameter_block
                command.push_constants = list(p._push_constants.items())
                command.source_resource = p._source_resource
                command.destination_resource = p._destination_resource
                command.copy_bytes = p._copy_bytes
                pd.commands = [command]

            pass_list.append(pd)
        desc.passes = pass_list

        desc.output_texture = self._output
        # RenderStack can remove/re-append passes while composing effects.
        # Resolve the boundary from the finished topology, not an earlier size.
        boundary = next((i for i, p in enumerate(self._passes)
                         if p._name == "_DisplayEncode"), None)
        desc.linear_output_texture = (self._linear_output or "") if boundary is not None else ""
        desc.linear_output_pass_count = boundary if boundary is not None else 0
        desc.msaa_samples = self._msaa_samples
        desc.temporal_jitter = self._temporal_jitter
        return desc

    # ---- Debug ----

    def get_debug_string(self) -> str:
        """Get a human-readable representation of the graph."""
        lines = [f"RenderGraph '{self._name}':"]
        lines.append(f"  Textures ({len(self._textures)}):")
        for tex in self._textures:
            tag = " [camera_target]" if tex.is_camera_target else ""
            samples = "frame" if tex.samples == 0 else str(tex.samples)
            lines.append(f"    - {tex.name} ({tex.format.name}, samples={samples}){tag}")

        lines.append(f"  Buffers ({len(self._buffers)}):")
        for buffer in self._buffers:
            lines.append(f"    - {buffer.name} ({buffer.byte_size} bytes, usage={buffer.usage})")

        lines.append(f"  Passes ({len(self._passes)}):")
        for i, p in enumerate(self._passes):
            lines.append(f"    [{i}] {p._name} [{p._pass_type}] -> {p._action}")
            if p._reads:
                lines.append(f"          reads: {', '.join(p._reads)}")
            if p._write_colors:
                for slot, name in sorted(p._write_colors.items()):
                    lines.append(f"          writes color[{slot}]: {name}")
            if p._write_depth:
                lines.append(f"          writes depth: {p._write_depth}")
            if p._resolve_color:
                lines.append(f"          resolves color[0]: {p._resolve_color}")
            if p._buffer_accesses:
                accesses = ", ".join(f"{name}:{access}" for name, access in p._buffer_accesses)
                lines.append(f"          buffers: {accesses}")

        if self._output:
            lines.append(f"  Output: {self._output}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<RenderGraph '{self._name}' "
                f"passes={len(self._passes)} textures={len(self._textures)} "
                f"buffers={len(self._buffers)}>")
