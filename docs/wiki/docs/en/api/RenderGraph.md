# RenderGraph

<div class="class-info">
class in <b>Infernux.rendergraph</b>
</div>

## Description

A declarative render graph that defines texture resources and render passes.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `RenderGraph.__init__(name: str = ..., output_samples: int = 0) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| name | `str` | The name of this render graph. *(read-only)* |
| pass_count | `int` | Number of render passes in the graph. *(read-only)* |
| texture_count | `int` | Number of texture resources in the graph. *(read-only)* |
| buffer_count | `int` | Number of buffer resources in the graph. *(read-only)* |
| topology_sequence | `List[Tuple[str, str]]` | Ordered list of (pass_name, type) entries defining the execution order. *(read-only)* |
| injection_points | `list` | List of injection points for pass extension. *(read-only)* |
| effect_stages | `List[EffectStage]` | Pipeline-declared user attachment stages in topology order. *(read-only)* |
| current_effect_resources | `Mapping[str, TextureHandle]` |  *(read-only)* |
| current_pass_result | `PassResult | None` |  *(read-only)* |
| geometry_buffer_requirements | `` |  *(read-only)* |
| pass_results | `Mapping[str, PassResult]` |  *(read-only)* |
| latest_pass_result | `PassResult | None` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_temporal_jitter(enabled: bool = True) → None` |  |
| `set_msaa_samples(samples: int) → int` | Set screen MSAA preference; return the effective Camera target sample count. |
| `create_texture(name: str, format: Format = ..., camera_target: bool = ..., size: Optional[Tuple[int, int]] = ..., size_divisor: int = ..., samples: Optional[int] = ...) → TextureHandle` | Declare a transient texture resource in the render graph. |
| `get_texture(name: str) → Optional[TextureHandle]` | Get a texture handle by name, or None if not found. |
| `import_texture(name: str, texture: RenderTexture | InxTexture, attachment: str = 'color') → TextureHandle` | Import a persistent render target attachment or a GUID-backed 2D/3D texture asset. Asset textures are sampled read-only and track reimports without rebuilding the scene. |
| `create_temporal_history(name: str, format: Format = Format.RGBA16_SFLOAT, size: Optional[Tuple[int, int]] = None, size_divisor: int = 0) → Tuple[TextureHandle, TextureHandle]` |  |
| `name_scope(prefix: str) → AbstractContextManager[RenderGraph]` |  |
| `effect_resources(resources: Mapping[str, TextureHandle]) → AbstractContextManager[RenderGraph]` |  |
| `pass_result(result: PassResult) → AbstractContextManager[RenderGraph]` |  |
| `replace_current_pass_result(result: PassResult) → None` |  |
| `resolve_effect_route_policy(stages)` |  |
| `create_buffer(name: str, byte_size: int, storage: bool = ..., indirect: bool = ..., transfer_source: bool = ..., transfer_destination: bool = ...) → BufferHandle` | Declare a transient buffer resource in the render graph. |
| `get_buffer(name: str) → Optional[BufferHandle]` | Get a buffer handle by name, or None if not found. |
| `has_pass(name: str) → bool` | Check if a render pass with the given name exists. |
| `has_injection_point(name: str) → bool` | Check if an injection point with the given name exists. |
| `has_effect_stage(stable_id: str) → bool` | Check for an exact declared stage ID. |
| `injection_point(name: str, display_name: str = ..., resources: Optional[set] = ...) → None` | Declare an injection point where external passes can be inserted. |
| `effect_stage(stable_id: str, scope: EffectScope | str = ..., display_name: str = ..., inputs: Optional[set[str]] = ..., outputs: Optional[set[str]] = ..., capabilities: Optional[set[str]] = ...) → EffectStage` | Declare a stable user-facing RenderEffect attachment stage. |
| `effects(stable_id: str) → EffectStage` | Pipeline-author shorthand for ``effect_stage``. |
| `screen_ui_section(resources: set | None = ..., world_ui_layer_mask: int = ...) → None` | Declare a screen UI section in the graph topology. |
| `set_geometry_buffer_requirements(requirements) → None` |  |
| `require_geometry_buffers(requirements) → None` |  |
| `needs_geometry_buffer(semantic: str) → bool` |  |
| `publish_pass_result(source: str, buffers, materialize = ...) → PassResult` |  |
| `derive_pass_result(source: str, parent: PassResult, overrides) → PassResult` |  |
| `write_buffer(source: str, parent: PassResult, name: str, texture: TextureHandle) → PassResult` |  |
| `get_pass_result(source: str) → PassResult | None` |  |
| `camera_ui_section(resources: set | None = ..., world_ui_layer_mask: int = ...) → None` | Draw Camera UI and declare the after-camera-UI effect stage. |
| `screen_ui_overlay_section(resources: set | None = ...) → None` | Encode for display, draw Screen UI, and declare its effect stage. |
| `add_pass(name: str) → RenderPassBuilder` | Add a new render pass to the graph. |
| `add_copy_pass(name: str) → RenderPassBuilder` | Add a transfer-domain texture or buffer copy pass. |
| `add_present_pass(name: str) → RenderPassBuilder` | Add a final graph export pass. |
| `remove_pass(name: str) → RenderPassBuilder | None` | Remove a render pass by name. |
| `append_pass(builder: RenderPassBuilder) → None` | Append an existing RenderPassBuilder to the graph. |
| `set_output(texture: str | TextureHandle) → None` | Set the final output texture of the render graph. |
| `validate_no_ip_before_first_pass() → None` | Validate that no user extension point appears before the first pass. |
| `get_debug_string() → str` | Return a human-readable summary of the graph for debugging. |
| `build() → RenderGraphDescription` | Compile the graph into a RenderGraphDescription for the backend. |

<!-- USER CONTENT START --> public_methods

`InxTexture` imports require a published asset GUID and explicit `2d` or `3d`
metadata. The shader resource dimension must match the asset dimension. Imported
assets cannot be graph outputs, copy destinations, color attachments, depth
attachments, or resolve targets. A cold asset remains pending without blocking
the editor frame; the graph retries publication when the asset is ready.

A fullscreen shader that declares `Requires [Lighting]` receives the current
camera/View's lighting UBO, scoped Forward+ light list, shadow atlas, and camera
position through descriptor set 1. The shadow texture still has to be declared
as an explicit `shadow_map` pass dependency so graph scheduling remains visible.
In a declarative Deferred route, each GBuffer geometry pass declares
`set_texture("shadowMap", shadow_map)` when shadows are enabled. This publishes
the same View's shadow image to the per-view lighting descriptor and keeps the
shadow caster pass ordered before lighting. A route without shadows declares no
`shadowMap` binding.

In the default Deferred pipeline, `gbuffer_normal` is private: RGB holds the
encoded world normal and alpha holds smoothness for deferred lighting. When a
consumer requests the public `normal` buffer, the GBuffer result publishes a
separate full-resolution RGBA16 texture with encoded normal RGB and alpha 1
for visible Deferred-compatible geometry, 0 elsewhere. After the Forward+
opaque pass, the `opaque_lighting` result copies the current public normal
(including `after_gbuffer` effect writes) and overlays visible
Deferred-unsupported geometry against the final read-only depth. Later stages
inherit that result. Effects that replace `normal` must preserve its coverage
alpha contract.

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example
> **Example status:** No curated example has been verified for this symbol in 0.4.0. Use the signatures above; do not infer behavior from similarly named APIs in other engines.
<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
