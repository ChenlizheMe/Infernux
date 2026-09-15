<!-- language:en -->

<span class="mini-tag">Custom Rendering · Chapter 8</span>

# RenderGraph for specialized pipelines

`RenderPipeline.define()` is the high-level authoring API from Chapter 7. The base `define_topology()` implementation compiles that DSL into explicit resources and passes. Override `define_topology(graph)` when the pipeline needs to author those details directly; the override receives a `RenderGraph`, and `define()` is no longer called.

The low-level API exposes more topology and more obligations: resource names, pass declarations, reads and writes, sample counts, EffectStage contracts, and final output all become part of the pipeline.

<div class="learn-article-toc"><strong>In this chapter</strong><a href="#when">Choose the API level</a><a href="#host-matrix">Host capability matrix</a><a href="#minimal-graph">Complete RenderStack graph</a><a href="#providers">Provider contracts</a><a href="#pass-results">PassResult and handle lifetime</a><a href="#resource-usage">Resource usage and MSAA</a><a href="#current-boundaries">Current boundaries</a><a href="#debugging">Validation and recovery</a></div>

<div class="learn-note"><strong>First-pass finish line.</strong><p>Use the complete <code>BaseColorPresentPipeline</code>, select it through a RenderStack host, and confirm that its color reaches the Game view and Present without validation errors. This chapter is the low-level path; Provider details, handle lifetime, resource usage, and MSAA are diagnostic references after the complete graph runs.</p></div>

<figure class="learn-figure">
  <img src="../assets/learn/rendergraph-debugger.webp" alt="conceptual pass and resource inspection layout" loading="lazy" decoding="async">
  <figcaption>Concept only, with the top lane showing authored pass order. The native compiler derives dependencies, barriers, and dead-pass removal from declared accesses. This is not an Editor capture; the current Editor has no graphical RenderGraph debugger.</figcaption>
</figure>

## Choose the API level {#when}

Use `define(pipeline)` for frame policy, opaque and transparent Queue routes, Forward/Forward+/Deferred selection, layers, sky, and Effect mount points. Use `define_topology(graph)` for a custom target layout, a nonstandard GBuffer, a pass dependency outside the route DSL, or a new source-scoped semantic buffer.

Keep the ownership chain in mind: the scene owns a RenderStack, the RenderStack owns the selected RenderPipeline and the ordered EffectSlots, the pipeline records topology on the Python RenderGraph builder, and `graph.build()` serializes that topology into a `RenderGraphDescription`. The native engine then compiles the description into a per-camera graph and reuses it across steady frames, keyed by the description's `source_revision`. The pipeline authors the topology; the engine executes it.

The two methods operate at different levels:

- `define(pipeline)` receives a `PipelineBuilder`. It validates domain ownership and compiles routes, intermediate images, resolves, and composition passes.
- `define_topology(graph)` receives a `RenderGraph`. The pipeline creates textures and buffers, declares passes, publishes results, and chooses the output.

Choose one method for each pipeline class. The built-in pipelines remain useful references for low-level code because they currently implement `define_topology(graph)` directly.

## RenderStack and standalone capability matrix {#host-matrix}

`RenderPipeline.render(context, camera)` is the standalone host. RenderStack has a separate build path that installs Effect callbacks, sets the pipeline's private defining-graph state, completes the standard tail, and applies failure recovery. Those differences are observable in the current source:

| Capability in an overridden `define_topology(graph)` | Through RenderStack | Standalone `RenderPipeline.render()` |
| --- | --- | --- |
| `graph.create_texture()`, pass builders, `graph.set_output()` | Supported | Supported |
| `self.require_buffer()` | Supported while RenderStack calls the override | **Unsupported currently:** raises because `_defining_graph` was not set |
| `self.publish_result()` and `self.write_buffer()` | Supported while RenderStack calls the override | **Unsupported currently:** same `_defining_graph` limitation |
| Direct `graph.require_geometry_buffers()`, `graph.publish_pass_result()`, `graph.write_buffer()` | Supported | Supported |
| `@geometry_buffer` plus `self.geometry_stage()` | Supported; RenderStack also adds mounted Effect requirements | Supported only when requirements are set directly on `graph` |
| Mounted RenderStack Effects and stage-local resource buses | Compiled at declared stages | No RenderStack instance is present, so stages are declarations only |
| Missing standard post-process and Screen UI tail | RenderStack appends the safety net | Pipeline must call the needed section helpers itself |
| Failed rebuild | Keeps a previous valid graph or uses the documented first-build Editor fallback | No fallback cache; the build exception leaves `_standalone_desc` unset and the next call retries |

The three `self.*` helpers are not promised for a standalone override in the current implementation. A standalone author can use the direct `graph.*` result methods. Setting `self._defining_graph` manually relies on private state and is excluded from the supported contract. The complete example below intentionally targets RenderStack.

In both hosts, return from `define_topology()` after recording declarations; the host calls `graph.build()`. Call `build()` directly only in an isolated topology test like the verification used for this chapter.

## Complete RenderStack graph: provider to present {#minimal-graph}

Save this class under `Assets`, select **Base Color Present** on the scene RenderStack, and use an active Camera with an opaque MeshRenderer. It performs a depth prepass, lazily materializes a custom base-color Provider, publishes that texture through a `PassResult`, copies it to the camera target, runs the canonical UI/display tail, and presents the result.

```python
import infernux as inx


class BaseColorPresentPipeline(inx.renderstack.RenderPipeline):
    name = "Base Color Present"

    @inx.renderstack.geometry_buffer("preview_color", dependencies={"depth"})
    def provide_preview_color(self, context):
        target = context.graph.create_texture(
            f"{context.source}_preview_color",
            format=inx.rendergraph.Format.RGBA16_SFLOAT,
        )
        with context.graph.add_pass(
            f"{context.source}_preview_color"
        ) as render_pass:
            render_pass.read(context.sample("depth"))
            render_pass.write_color(target)
            render_pass.set_clear(color=(0.0, 0.0, 0.0, 0.0))
            render_pass.draw_renderers(
                queue_range=context.queue_range,
                sort_mode=context.sort_mode,
                material_pass="base_color",
            )
        return target

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        depth = graph.create_texture(
            "depth", format=inx.rendergraph.Format.D32_SFLOAT
        )

        with graph.add_pass("DepthPrepass") as render_pass:
            render_pass.write_depth(depth)
            render_pass.set_clear(depth=1.0)
            render_pass.draw_renderers(
                queue_range=(0, 2500),
                sort_mode="front_to_back",
                material_pass="depth",
            )

        requested = self.require_buffer("preview_color")
        opaque = self.geometry_stage(
            graph,
            "opaque",
            buffers={"depth": depth},
            queue_range=(0, 2500),
        )
        preview = self.sample_buffer(opaque, requested)

        color = graph.create_texture("color", camera_target=True)
        with graph.add_pass("CopyToCamera") as render_pass:
            render_pass.set_texture("_SourceTex", preview)
            render_pass.write_color(color)
            render_pass.fullscreen_quad("Fullscreen Blit")

        camera_result = self.write_buffer(
            opaque,
            "color",
            color,
            source="camera",
        )

        with graph.pass_result(camera_result):
            graph.screen_ui_section(resources={"color", "depth"})

        with graph.add_present_pass("Present") as present_pass:
            present_pass.present(color)
```

`present(color)` is a typed terminal action and also calls `set_output(color)`. A graph may use `set_output()` without a Present pass, but this example makes the camera-target/export boundary visible. `graph.build()` also chooses the first camera target when no explicit output exists; production pipelines should express the intended output directly.

To verify the example, check that the camera shows unlit base color, the RenderStack topology contains the standard tail and **Present**, and the Console has no graph validation error. Temporarily add `print(graph.get_debug_string())` as the final line of `define_topology()` to record the authored resources, actions, accesses, and output, then remove the print after diagnosis.

Logical names are graph-wide resource identities. Reusable fragments can use `graph.name_scope()` to keep generated names unique. Pass order is recorded as authored, while declared accesses give the native compiler dependency and transition information. The native compiler can cull work that has no path to an output or explicit side effect, so authored order alone does not prove execution.

`screen_ui_section()` places Camera UI, post-process points, display encoding, and Screen UI at that location. A graph built through RenderStack also receives the required post-process points and Screen UI tail when they are missing. Calling the method explicitly keeps their position clear. Standalone use of `RenderPipeline.render()` has no RenderStack safety net, so the pipeline must complete its own output contract.

The standard tail expects the logical `color` resource. A specialized RenderStack pipeline can set another final output, though it still needs a valid `color` path for the standard UI and display-encoding sections.

## Provider contracts {#providers}

Geometry-buffer providers are methods registered on a pipeline class with `@geometry_buffer`. The registration key is `(semantic, phase)`; the default phase is `opaque`. Dependencies are semantic names, and the compiler orders providers topologically.

```python
import infernux as inx


class ObjectIndexPipeline(inx.renderstack.RenderPipeline):
    name = "Object Index"

    @inx.renderstack.geometry_buffer("object_index", dependencies={"depth"})
    def provide_object_index(self, context):
        target = context.graph.create_texture(
            f"{context.source}_object_index",
            format=inx.rendergraph.Format.RG32_UINT,
        )
        with context.graph.add_pass(
            f"{context.source}_object_index"
        ) as render_pass:
            render_pass.read(context.sample("depth"))
            render_pass.write_color(target)
            render_pass.draw_renderers(
                queue_range=context.queue_range,
                sort_mode=context.sort_mode,
                material_pass="picking",
            )
        return target
```

A Provider receives a `GeometryBufferProviderContext`. It may read `context.graph`, `context.source`, `context.phase`, `context.queue_range`, `context.msaa_samples`, `context.sort_mode`, `context.clear`, and already available semantic textures through `context.sample()`. It must return a non-null graph `TextureHandle`; `geometry_stage()` publishes that handle under the decorator's semantic. Dependencies name other semantics that must already exist or have a Provider in the same phase.

A derived class can replace a built-in provider by declaring the same semantic and phase. Two providers for the same key in one class are ambiguous and rejected. Missing dependencies and dependency cycles also fail topology construction with the source and dependency chain in the error. Provider methods run during each topology build when their semantic is requested; the API defines no cross-graph Provider instance cache. Keep build-local handles in the context and keep persistent CPU policy on the pipeline instance.

During `define_topology()`, call `self.require_buffer("object_index")` before the relevant `geometry_stage()`. The stage starts from its supplied buffers, runs only the providers needed by current requirements, and returns a `PassResult`. Effects mounted in RenderStack contribute their declared geometry requirements before the pipeline topology is built, so unused built-in providers such as normal or motion remain unmaterialized.

```python
requested = self.require_buffer("object_index")
result = self.geometry_stage(
    graph,
    "opaque",
    buffers={"color": color, "depth": depth},
    queue_range=(0, 2500),
)
object_index = self.sample_buffer(result, requested)
```

`require_buffer()` is valid only while RenderStack or the base DSL implementation has set the defining graph. The returned `BufferHandle` is a semantic request from `Infernux.renderstack`; it is separate from the transient GPU `BufferHandle` returned by `graph.create_buffer()`. Standalone overrides use `graph.require_geometry_buffers({"object_index"})`, then pass the same graph into `geometry_stage()`.

## PassResult, handles, and native actions {#pass-results}

A `PassResult` is a source-scoped map from semantic names such as `color`, `depth`, `normal`, and `motion` to texture handles. `source` must be unique within one graph build. The graph assigns an increasing `revision` each time it publishes or derives a result.

```python
before = self.publish_result(
    "opaque",
    {"color": color, "depth": depth},
)

copied = graph.create_texture(
    "post_color",
    format=inx.rendergraph.Format.RGBA16_SFLOAT,
)
with graph.add_pass("CopyColor") as render_pass:
    render_pass.set_texture("_SourceTex", before.sample("color"))
    render_pass.write_color(copied)
    render_pass.fullscreen_quad("Fullscreen Blit")

after = self.write_buffer(
    before,
    "color",
    copied,
    source="post_copy",
)
```

`write_buffer()` derives a result with one semantic replaced. The parent still refers to the earlier texture, so later topology can deliberately sample either revision. The revision number is local to the graph build: it expresses publication order, not a frame number, persistent asset ID, or mutable GPU-resource version.

Lazy geometry providers may add a missing semantic to the result that owns them. Once a write derives a new result, earlier semantic bindings stay intact.

`publish_pass_result()` accepts only semantic names mapped to graph `TextureHandle` objects, and every `source` must be unique in that graph build. `PassResult.sample()` returns the logical handle; `snapshot` returns a read-only copy of the current semantic mapping. Result publication does not allocate, copy, or mutate a GPU image. A pass declaration must still write the texture, and a downstream pass must declare its read.

Texture and GPU Buffer handles are lightweight logical-name records owned by one builder run. Do not retain them on the pipeline instance, reuse them after a rebuild, or pass them into another graph. The Python handle type does not carry a graph ID, so a same-name cross-graph mistake can evade early identity checks. The resulting `RenderGraphDescription` contains names and resource descriptions; the native per-camera graph creates the actual resources.

All camera targets in one graph alias the camera's physical color target. Declaring more than one emits a warning. A persistent `inx.RenderTexture` has separate ownership: assign it to `Camera.target_texture`, `UIImage.texture`, or a material texture binding, and use `graph.import_texture()` for explicit graph access. Rebuilding a graph does not destroy the resource. These runtime references do not fabricate asset GUIDs or become saved texture assets.

Choose either fixed pixels or a scale relative to the Game render resolution. Create the resource on the engine thread, for example in a component's `start()`:

```python
PixelFormat = inx.rendergraph.Format

fixed = inx.RenderTexture(641, 401, depth_format=PixelFormat.D32_SFLOAT)
half_size = inx.RenderTexture(
    scale=(0.5, 0.5),
    format=PixelFormat.RGBA16_SFLOAT,
    depth_format=PixelFormat.D32_SFLOAT,
    samples=4,
)
camera.target_texture = half_size
image.texture = half_size
```

The relative target follows **Game render pixels**, not the Game panel's display zoom, desktop DPI, Scene View size, or the dimensions of the last camera drawn. At a 641×401 Game resolution, `half_size.width, half_size.height` is `(321, 201)`: fractional pixels round up. Both factors must be positive and finite; factors above one request supersampling. Changing the Game resolution updates the existing owner and its Camera, material and UI bindings together. Do not call `resize()` every frame.

`fixed.scale` is `None` and `fixed.resize(width, height)` changes its pixel dimensions. A relative target's `scale` is read-only and pixel `resize()` is rejected. Changing size mode, scale, format, depth or MSAA means creating a new resource and replacing the reference. Allocation failures leave the old allocations intact and report the error; they do not silently select a smaller size or another format. `sampled_depth=True` explicitly enables depth texture sampling; enable it when the Camera pipeline uses TAA or another depth-sampling effect. A graph that samples an attachment without this declaration is rejected before submission. `storage=True` requests storage-image use, not ordinary raster rendering.

This reference-size concept is comparable to [Unity's RTHandle scale allocation](https://docs.unity.cn/Packages/com.unity.render-pipelines.core%4016.0/manual/rthandle-system-using.html), but the public Infernux type remains `RenderTexture`. Infernux allocates the requested rounded size on each resolution change, including shrink; it does not promise Unity's maximum-reference-size allocation policy.

For a reusable asset, choose **Create > Render Texture** in Project. New assets have a D32 depth attachment, ready for Camera assignment; color-only uses can explicitly disable depth in the Inspector. The Inspector edits the same size, format, MSAA and usage description; edits use ordinary autosave and Undo/Redo. Selecting the asset does not allocate a GPU image. The importer creates `Library/Artifacts/RenderTexture/<guid>.inxrtex`; Cook includes this binary description, not the source JSON or rendered pixels.

Expose an ordinary serialized field and assign that asset in the Inspector:

```python
import infernux as inx

class Monitor(inx.InxComponent):
    output: inx.RenderTexture

    def start(self):
        self.game_object.get_component(inx.Camera).target_texture = self.output
```

Enable a depth format on targets used by Cameras. `inx.RenderTexture.load("Assets/Rendering/Monitor.rendertexture")` and `load_by_guid(guid)` resolve the same GPU owner as serialized references. Reimporting an asset updates existing users rather than replacing their references; removing depth while a Camera owns the target is rejected. Fixed/relative size changes are authored on the asset, whereas runtime `resize()` does not write the source file. Only imported targets have a `guid` and `file_path`; runtime-created targets remain transient.

You can also drag the asset directly into **Camera → Target Texture**, without a script. This slot is saved with the scene and works in Edit mode and Play mode. Removing the asset preserves a missing reference instead of redirecting that camera to the screen; restoring the same GUID reconnects its output. The source file is not read during Player scene decoding: the camera keeps its GUID and the renderer prepares the imported binary description.

Material texture fields also accept imported RenderTextures. Drag the asset into a material's texture slot, or call `material.set_texture("texSampler", target)`: an imported target saves its GUID; an anonymous runtime target only overrides the live binding. Reimport updates existing consumers, including runtime material copies. Deleting a target disconnects those bindings without erasing the saved GUID; restoring the asset reconnects them.

Assign that material to a MeshRenderer or to `UIImage.material` to display the camera output without CPU image readback. This Project → Camera → Material workflow follows [Unity's Render Texture example](https://docs.unity3d.com/6000.0/Documentation/Manual/output-to-render-texture.html).

For an image without a custom material, drag a Texture or RenderTexture asset into **UIImage → Texture**. This single slot saves the concrete asset type and GUID with the scene; it does not serialize rendered pixels. `image.texture = imported_target` uses the same reference. Assigning an anonymous runtime RenderTexture temporarily overrides the saved image; assigning `None` removes that override. The Inspector's Clear action edits the saved slot through scene Undo. Old `texture_path` scene fields are converted at load time and are no longer written. The Windows Player uses the cooked GUID-addressed description from `Content.inxpkg`; it does not need the original `.rendertexture` file. Camera, material and image consumption has been verified together in a standalone Windows build. Other-platform acceptance remains pending.

Camera matrices and material/draw matrix inputs share NumPy `(4, 4)` indexing: `[row, column]`. You can pass a camera matrix directly, without flattening or transposing it. For example, a planar-mirror shader can declare a `Mat4 reflectionVP` property and receive its camera transform as a per-renderer parameter:

```python
view_projection = reflection_camera.projection_matrix @ reflection_camera.view_matrix
mirror_renderer.set_parameter("reflectionVP", view_projection)
```

Use `Material.set_matrix(name, matrix)` to edit the material itself, or `DrawParameterBlock.set_matrix(name, matrix)` for an explicit draw. Renderer overrides leave the shared material unchanged. These setters copy the values; later edits to the NumPy array do not update the material. Existing flat 16-number inputs and saved matrix documents remain column-major. Matrix transfer does not change camera clip-space conventions.

`graph.create_temporal_history(name, format=..., size=... / size_divisor=...)` creates a per-view, single-sample read/write pair. The first read after creation or invalidation is zero; write the output on every execution. The renderer owns ping-pong and resource retirement. History allocation does not enable camera jitter: TAA requests it separately with `graph.set_temporal_jitter()`. Do not write the previous-frame input or read and write the same image in one pass.

Use `camera.reset_history()` after a small teleport or another discontinuous change that automatic camera-cut detection cannot identify. It invalidates the existing temporal resources and previous camera matrix on the next render of each view using that Camera. It does not clear another Camera's history, change the Transform/projection, modify the scene document, or rebuild the graph. Repeated calls before rendering coalesce into one reset per view; accumulation then resumes. This is an explicit camera-history operation, comparable in purpose to [URP's resetHistory](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@17.0/api/UnityEngine.Rendering.Universal.UniversalAdditionalCameraData.html#UnityEngine_Rendering_Universal_UniversalAdditionalCameraData_resetHistory), without requiring an additional Camera component.

A pass builder records one typed action such as `draw_renderers()`, `fullscreen_quad()`, `copy_texture()`, or `present()`. Calling an action method replaces the previous action on that pass. Python receives no native pass callback, command encoder, Vulkan handle, or resolved GPU resource. `graph.build()` serializes declarations into `GraphPassDesc` and `GraphCommandDesc`; `context.apply_graph()` hands that description to the native compiler and executor.

## Resource usage and MSAA {#resource-usage}

The Python texture API currently derives usage from pass declarations:

- `write_color()` and `write_depth()` declare attachment writes.
- `read()` declares a texture dependency; `set_texture()` also adds the read and records the shader binding.
- `write_resolve()` declares a color resolve target.
- `copy_texture()` declares transfer source and destination access for a copy pass.

`create_texture()` therefore has no public `usage=` parameter. Every actual use still needs a matching pass declaration. A sampled depth texture must appear as a read or sampler binding; an attachment declaration alone does not make the sampling dependency visible to the graph.

Transient GPU buffers use explicit creation flags:

```python
draw_data = graph.create_buffer(
    "draw_data",
    64 * 1024,
    storage=True,
    indirect=True,
    transfer_destination=True,
)
```

`read_buffer()` and `write_buffer()` validate storage, indirect, or transfer access against those flags. `copy_buffer()` adds transfer source/destination flags to its two handles. These declarations describe access and synchronization; an executable pass action still has to use the resource.

`samples = graph.set_msaa_samples(1|2|4|8)` declares the screen preference and returns the effective sample count. A fixed `Camera.target_texture` owns its sampling contract and takes precedence. Build MSAA-dependent attachments and resolve passes using the returned value, not a hard-coded count; do not rewrite the authored pipeline parameter for each Camera. RenderStack and standalone pipelines cache separate graph variants by this contract. `set_msaa_samples(0)` delegates screen sampling to the native setting; it is not suitable for choosing explicit resolve topology. Camera targets and scene-sized depth textures default to inherited `samples=0`; other transient textures default to one sample. Raster color and depth attachments must agree.

Use `write_resolve()` when a multisampled color result must become a single-sample texture:

```python
samples = graph.set_msaa_samples(4)
resolved = graph.create_texture(
    "route_color", format=inx.rendergraph.Format.RGBA16_SFLOAT, samples=1,
)
color = resolved
if samples > 1:
    color = graph.create_texture(
        "route_msaa", format=inx.rendergraph.Format.RGBA16_SFLOAT, samples=samples,
    )
depth = graph.create_texture(
    "depth", format=inx.rendergraph.Format.D32_SFLOAT, samples=samples,
)

with graph.add_pass("Route") as render_pass:
    render_pass.write_color(color)
    render_pass.write_depth(depth)
    if samples > 1:
        render_pass.write_resolve(resolved)
    render_pass.draw_renderers(queue_range=(0, 2500))
graph.set_output(resolved)
```

The pass must have exactly one color output at slot `0`. The source must be multisampled; the target must be a transient, single-sample color texture with matching format and extent. The current Python API has no depth-resolve operation.

## Raw object-data masks {#object-data-mask-en}

A replacement material can write data instead of surface color. For example, a project outline can store a logical owner ID, positive eye depth, an invalid-state bit, and two flags. Register the renderers through `RendererSelection`; the engine still draws their current geometry. Group parts under the same Rigidbody in project code, rather than using the scene's picking IDs as gameplay group IDs.

Create a fragment shader asset named `ObjectDataMask.frag` and assign it to a material in the Inspector, with the built-in `Standard` vertex shader:

```glsl
#version 450
ShaderInfo {
    Name "ObjectDataMask"
    ShadingModel Unlit
    CastShadows Off
    Capabilities [ForwardOnly, NoDepthPass, NoPicking, NoMotionVectors, NoNormalPass, NoBaseColorPass]
    Properties {
        Float ownerId = 1.0
        Float invalid = 0.0
        Float flags = 0.0
    }
}
void main() {
    outColor = vec4(material.ownerId, v_ViewDepth, material.invalid, material.flags);
}
```

The capability declarations are intentional: this raw `main()` writes an application data format, not the engine's generated shadow, picking, motion, normal or albedo formats. `NoDepthPass` disables the generated depth-only variant; it does **not** disable the mask draw's own depth test/write. If authoring a complete material document through MCP, include the declared properties and their defaults as well as the shader references; changing just the shader name is not the Inspector's shader-assignment operation.

Use a single-sample `RGBA32_SFLOAT` mask and a matching independent `D32_SFLOAT` attachment, such as `owner_mask_depth`. Reserve zero for the cleared background and cap dense owner IDs at `2**24 - 1`; half-float masks do not preserve IDs at that scale. Sample IDs/flags with `texelFetch`, not filtered color sampling or multisample averaging. In this layout, `flags = int(hatched) + 2 * int(depth_tested)`. Set these floats with `DrawParameterBlock` before recording each renderer; the block is captured per draw.

Draw the mask after opaque geometry using its independent depth, then composite it over scene color using a copied scene depth. This preserves selected objects behind unselected walls while still resolving occlusion within the selected set. It does not automatically reproduce source-material alpha clipping or vertex-shader deformation: the replacement shader must express those when the effect requires them. GPU mesh-buffer deformation already uses the current shared geometry.

For a clipped surface, declare the coverage parameters in both shaders and copy the source material's values into each selected draw. Use the same UV predicate and face-culling policy in the surface and mask. When switching to a solid material, explicitly restore the mask's solid-coverage values; a reused parameter block otherwise keeps the previous draw's values. This is a project-authored material contract, not automatic shader translation.

For automatic occlusion hatching, a project can compare its object depth with the farthest scene depth in a small cross-shaped neighborhood. This suppresses thin foreground coverage without removing hatching behind a broad wall. Keep the sampling radius in pixels and the absolute/relative depth bias as project parameters; zero radius gives a useful per-pixel comparison. Apply this tolerance only to the automatic hatching decision, not to explicitly depth-tested outlines. Alpha-blended surfaces normally do not contribute opaque depth; an alpha-cutout occluder must discard its holes before writing depth. A selected glass object's geometric silhouette and a selected leaf's clipped silhouette are different authored mask policies.

## World UI at different render stages {#world-ui-stages-en}

`draw_world_ui(layer_mask=...)` draws UI from selected GameObject layers. It intersects the unsigned 32-bit mask with the current Camera's culling mask; `0` draws none and the default `0xffffffff` includes all camera-visible layers. This serves the stage/layer-filtering role described in Unity's [Render Objects reference](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/renderer-features/renderer-feature-render-objects.html), without a separate UI renderer or Camera.

For labels that should stay outside post-processing, exclude their layer from ordinary World UI and draw that layer later:

```python
labels = 1 << 30  # Assign these labels' GameObjects to layer 30.
graph.camera_ui_section(world_ui_layer_mask=0xffffffff ^ labels)

# Add the project's post-processing passes here.

with graph.add_pass("LateLabels") as p:
    p.write_color("color").write_depth("depth")
    p.draw_world_ui(layer_mask=labels)
graph.screen_ui_overlay_section()  # Display encoding and screen overlay UI.
```

World UI always keeps its normal depth test, using the depth attachment you declare for that pass. Drawing later does not make a label visible through walls. A project may instead attach a compatible depth snapshot from an earlier stage; the snapshot must be produced explicitly. Keep masks disjoint to avoid drawing an element twice. `screen_ui_section(world_ui_layer_mask=...)` forwards the same filter to its ordinary World UI pass; predeclared passes retain their own settings. Screen-space UI is unaffected. RenderTexture dependencies are collected from the same filtered elements that are drawn.

## Fullscreen depth and blending {#fullscreen-raster-state-en}

These controls cover the roles of Unity's [ZTest](https://docs.unity3d.com/6000.0/Documentation/Manual/SL-ZTest.html), [ZWrite](https://docs.unity3d.com/6000.0/Documentation/Manual/SL-ZWrite.html), and straight-alpha [Blend](https://docs.unity3d.com/6000.0/Documentation/Manual/SL-Blend.html), expressed as graph-pass declarations. This is not the full ShaderLab render-state surface.

`fullscreen_quad()` replaces color by default, with depth testing and writing disabled. An overlay can instead use the existing attachments:

```python
import infernux as inx

graph.add_copy_pass("PreserveSceneDepth").copy_texture(scene_depth, previous_depth)
with graph.add_pass("DepthAwareOverlay") as p:
    p.write_color(scene_color).write_depth(scene_depth)
    p.set_texture("sceneDepthTex", previous_depth)
    p.fullscreen_quad(
        "MyDepthAwareOverlay",
        depth_test=inx.rendergraph.DepthCompare.ALWAYS,
        depth_write=True,
        alpha_blend=True,
    )
```

The scene pass must have already initialized `scene_color` and `scene_depth`. The depth copy must match its source's format, extent and sample count. An attachment cannot also be sampled by the same pass.

Fullscreen inputs declared as `Texture2DMS` or `Texture2DMSUInt` in ShaderInfo `Resources` receive the original multisampled image. Read it with `texelFetch(texture, pixel, sampleIndex)`; categorical owner IDs and flags must retain their individual samples, not be averaged. Ordinary `Texture2D` inputs receive resolved color or depth. This choice is reflected from the shader when the graph is built, and shader hot reload rebuilds the affected resolve topology. A `Texture2DMS` input cannot bind a single-sample image.

For a per-sample fullscreen effect, use `gl_SampleID` and `gl_SamplePosition`, write the result into a matching multisampled color target, then resolve the final colors. This requires device support for sample-rate shading. Enabling MSAA on the scene color alone does not anti-alias edges generated later by a single-sample shader. Use the effective count returned by `graph.set_msaa_samples(4)` to select the single-sample or multisampled shader when a Camera has a fixed RenderTexture target. Both shader entry points can share their effect logic through ShaderInfo `Imports`.

The root pipeline's viewport-sized `depth` is the conventional Camera depth attachment. Other depth texture names create independent images, even when their dimensions match the viewport; use separate names for an object's mask depth or a depth snapshot. Fixed-size, reduced-size and imported RenderTexture depth keep their own resource identity.

`depth_test` accepts `DepthCompare` values such as `LESS_EQUAL` or `ALWAYS`; `None` disables it. Depth writes require a comparison, including `ALWAYS` for unconditional writes. A fragment shader can write `gl_FragDepth` in the current Vulkan view's device-depth range (0–1), not linear eye depth. Assign it on every non-discarded path when overriding depth.

Alpha blending uses straight-alpha source-over: `src.rgb * src.a + dst.rgb * (1-src.a)` and `src.a + dst.a * (1-src.a)`. Do not pre-composite the background into the shader's output. Blending and depth-tested passes preserve earlier color unless explicitly cleared; new transient attachments require a clear or an earlier producer. Non-fullscreen consumers of multisampled color still require a resolved input, such as one produced with `write_resolve()`. These states belong to the ordinary graph; no effect-specific engine renderer is needed.

## Current boundaries {#current-boundaries}

The public Python RenderGraph currently exposes three pass types:

- `add_pass()` creates a raster pass for renderer draws, sky, Screen UI, or fullscreen work.
- `add_copy_pass()` creates a copy pass for `copy_texture()` or `copy_buffer()`.
- `add_present_pass()` exports a color texture with `present()`.

Compute dispatch is outside the current Python API. There is no `add_compute_pass()`, `dispatch()`, or Python `GraphPassType.COMPUTE`. Storage and indirect Buffer usage flags are available for resource contracts, though they do not add a dispatch or indirect-draw command.

Choose an available path by workload:

- For image-space math that maps cleanly to fragment work, use a Raster pass with `fullscreen_quad()` and declared texture inputs.
- For ordinary scene submission, use `draw_renderers()` and Material Queue filtering; the engine owns renderer batching and draw submission.
- For GPU particle simulation and GPU-driven particle drawing, use the Particle Graph subsystem, whose compute and indirect path is engine-owned.
- For a generic compute kernel, GPU-generated indirect draw, custom queue, or custom native resource import, the Python RenderGraph is currently the wrong extension surface. That work requires an engine-owned native feature and a new public binding/IR contract before a project pipeline can call it.

Transfer support is deliberately narrow. Texture copies require distinct transient textures with matching formats; camera targets are excluded. Buffer copies accept an optional byte count and cannot exceed the smaller buffer. Arbitrary blits, format conversion, queue selection, and custom transfer commands have no public Python builder entry point today.

To write a processed image back to the Camera and continue drawing later passes, use the built-in fullscreen shader instead of a copy pass. This also applies when the Camera outputs to a saved RenderTexture asset:

```python
with graph.add_pass("CommitGrade") as render_pass:
    render_pass.write_color(camera_color)
    render_pass.set_texture("_SourceTex", graded)
    render_pass.fullscreen_quad("Fullscreen Blit")
```

Here `graded` is a separately produced, single-sample color texture; `camera_color` is the graph's camera target. Do not sample and write the same image in this pass. Use `present()` when exporting the final image instead of continuing to draw into the Camera target.

## Validation, diagnostics, and recovery {#debugging}

`graph.build()` checks duplicate resource and pass names, missing resources, action/pass-type mismatches, Buffer usage, attachment formats, sample-count agreement, resolve contracts, extension-point placement, and final output. The native compiler then validates and schedules the resulting graph.

RenderStack rejects a failed topology edit and retains the last accepted graph with its paired effect bindings for that output-sample contract. The Inspector reports the error alongside its last accepted topology probe. If there is no accepted graph, construction fails explicitly in both Editor and Player; it does not substitute Default Forward. Repairing a watched pipeline file or changing a pipeline parameter invalidates the failed state. This author-edit transaction is not permission to sample an unwritten texture or silently reuse old camera output.

Standalone `RenderPipeline.render()` has no last-valid or Default Forward recovery. A failed `define_topology()` or `build()` leaves `_standalone_desc` unset, so the exception remains visible and a later render call retries. After an accepted code replacement, `dispose()` clears an older standalone description when the pipeline is retired.

There is currently no graphical RenderGraph debugger in the Editor. Use `graph.get_debug_string()` for a text summary of resources, pass actions, reads, writes, resolves, and output, then confirm behavior in both Editor and a build. This string describes one topology artifact, so it cannot distinguish camera-local native instances. For multi-camera runtime logs, include the camera identity available to your host, `context.graph_instance_id`, and `RenderGraphDescription.source_revision`; `graph_instance_id` distinguishes native graph instances while the source revision identifies the shared Python topology.

Before shipping a low-level pipeline, check these points:

1. Every consumed texture or Buffer has the corresponding read/access declaration and an upstream producer.
2. Attachment sample counts agree, and every sampled MSAA color path resolves at the intended point.
3. EffectStage input/output contracts match the semantic resources available at that source.
4. Each camera receives its own graph-backed runtime resources and camera-local light/shadow state.
5. The graph has one final color output, one display encode, and intentional Camera UI and Screen UI placement.

<!-- language:zh -->

<span class="mini-tag">自定义渲染 · 第 8 章</span>

# 面向特殊管线的 RenderGraph

`RenderPipeline.define()` 是第 7 章介绍的高层编写 API。基类的 `define_topology()` 会把这套 DSL 编译为显式资源与 Pass。需要直接编写这些细节时，可以覆盖 `define_topology(graph)`；覆盖后收到的是 `RenderGraph`，系统也不会再调用 `define()`。

进入低层 API 后，拓扑控制范围更大，相应契约也更多：资源名、Pass 声明、读写、采样数、EffectStage Contract 与最终输出都由管线负责。

<div class="learn-article-toc"><strong>本章内容</strong><a href="#when_1">选择 API 层级</a><a href="#host-matrix_1">Host 能力矩阵</a><a href="#minimal-graph_1">完整 RenderStack Graph</a><a href="#providers_1">Provider 契约</a><a href="#pass-results_1">PassResult 与 Handle 生命周期</a><a href="#resource-usage_1">资源 Usage 与 MSAA</a><a href="#current-boundaries_1">当前能力边界</a><a href="#debugging_1">校验与恢复</a></div>

<div class="learn-note"><strong>第一次阅读的完成点。</strong><p>使用完整的 <code>BaseColorPresentPipeline</code>，通过 RenderStack Host 选中它，确认颜色进入 Game 画面并完成 Present，且没有 Validation Error。本章属于低层路径；Provider 细节、Handle 生命周期、资源 Usage 与 MSAA 适合在完整 Graph 跑通后作为诊断资料查阅。</p></div>

<figure class="learn-figure">
  <img src="../assets/learn/rendergraph-debugger.webp" alt="Pass 与资源检查布局的概念示意图" loading="lazy" decoding="async">
  <figcaption>仅用于解释概念，顶行表示编写时的 Pass 顺序。Native Compiler 根据已声明的 Access 推导依赖、Barrier 与 Dead-pass Removal。本图不是 Editor 截图；当前 Editor 没有图形化 RenderGraph Debugger。</figcaption>
</figure>

## 选择 API 层级 {#when_1}

`define(pipeline)` 适合帧策略、不透明与透明 Queue 路由、Forward/Forward+/Deferred 选择、Layer、天空和 Effect 挂载点。`define_topology(graph)` 适合自定义 Target 布局、非标准 GBuffer、Route DSL 词汇之外的 Pass 依赖，以及新的源作用域语义 Buffer。

记住这条所有权链：场景拥有 RenderStack，RenderStack 拥有选中的 RenderPipeline 与有序 EffectSlot，管线把拓扑记录到 Python RenderGraph 构建器上，`graph.build()` 再把拓扑序列化成 `RenderGraphDescription`。原生引擎随后把描述编译成每相机的图，并在稳态帧中按描述的 `source_revision` 复用。管线负责编写拓扑，引擎负责执行。

两个方法处在不同层级：

- `define(pipeline)` 收到 `PipelineBuilder`，负责校验 Domain 所有权，并编译 Route、中间图像、Resolve 与合成 Pass。
- `define_topology(graph)` 收到 `RenderGraph`，由管线创建 Texture 与 Buffer、声明 Pass、发布 Result，并选定输出。

每个管线类选择一个方法。内置管线当前直接实现 `define_topology(graph)`，可作为低层代码参考。

## RenderStack 与 standalone 能力矩阵 {#host-matrix_1}

`RenderPipeline.render(context, camera)` 是 standalone Host。RenderStack 使用另一条构建路径，它会安装 Effect Callback、设置管线的私有 Defining Graph 状态、补全标准帧尾并执行失败恢复。当前源码中的差异如下：

| 覆盖 `define_topology(graph)` 后的能力 | 通过 RenderStack | Standalone `RenderPipeline.render()` |
| --- | --- | --- |
| `graph.create_texture()`、Pass Builder、`graph.set_output()` | 支持 | 支持 |
| `self.require_buffer()` | RenderStack 调用 Override 期间支持 | **当前不支持：** `_defining_graph` 未设置，会抛出异常 |
| `self.publish_result()` 与 `self.write_buffer()` | RenderStack 调用 Override 期间支持 | **当前不支持：** 受同一 `_defining_graph` 限制 |
| 直接调用 `graph.require_geometry_buffers()`、`graph.publish_pass_result()`、`graph.write_buffer()` | 支持 | 支持 |
| `@geometry_buffer` 与 `self.geometry_stage()` | 支持；RenderStack 还会加入已挂载 Effect 的需求 | 需要直接在 `graph` 上设置需求后使用 |
| 已挂载的 RenderStack Effect 与 Stage 局部 Resource Bus | 在声明位置编译 | 没有 RenderStack 实例，Stage 只保留声明信息 |
| 缺失的标准后处理与 Screen UI 帧尾 | RenderStack 会追加安全网 | 管线必须自行调用所需 Section Helper |
| 重建失败 | 保留上一份有效 Graph，或使用已说明的 Editor 首次构建回退 | 没有回退缓存；异常后 `_standalone_desc` 为空，下次调用重试 |

当前实现没有承诺三项 `self.*` Helper 可用于 standalone Override。Standalone 作者可以改用直接的 `graph.*` Result 方法。手动设置 `self._defining_graph` 会依赖私有状态，不属于受支持契约。下面的完整样例明确以 RenderStack 为 Host。

两种 Host 都要求 `define_topology()` 记录完声明后直接返回，由 Host 调用 `graph.build()`。只有独立拓扑测试才应直接调用 `build()`，例如本章样例采用的验证方式。

## 完整 RenderStack Graph：从 Provider 到 Present {#minimal-graph_1}

把这个类保存到 `Assets` 下，在场景 RenderStack 中选择 **Base Color Present**，并准备一台活动 Camera 和一个不透明 MeshRenderer。样例先执行 Depth Prepass，再惰性生成自定义 Base-color Provider，通过 `PassResult` 发布纹理，把结果复制到 Camera Target，执行标准 UI/显示帧尾，最后 Present。

```python
import infernux as inx


class BaseColorPresentPipeline(inx.renderstack.RenderPipeline):
    name = "Base Color Present"

    @inx.renderstack.geometry_buffer("preview_color", dependencies={"depth"})
    def provide_preview_color(self, context):
        target = context.graph.create_texture(
            f"{context.source}_preview_color",
            format=inx.rendergraph.Format.RGBA16_SFLOAT,
        )
        with context.graph.add_pass(
            f"{context.source}_preview_color"
        ) as render_pass:
            render_pass.read(context.sample("depth"))
            render_pass.write_color(target)
            render_pass.set_clear(color=(0.0, 0.0, 0.0, 0.0))
            render_pass.draw_renderers(
                queue_range=context.queue_range,
                sort_mode=context.sort_mode,
                material_pass="base_color",
            )
        return target

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        depth = graph.create_texture(
            "depth", format=inx.rendergraph.Format.D32_SFLOAT
        )

        with graph.add_pass("DepthPrepass") as render_pass:
            render_pass.write_depth(depth)
            render_pass.set_clear(depth=1.0)
            render_pass.draw_renderers(
                queue_range=(0, 2500),
                sort_mode="front_to_back",
                material_pass="depth",
            )

        requested = self.require_buffer("preview_color")
        opaque = self.geometry_stage(
            graph,
            "opaque",
            buffers={"depth": depth},
            queue_range=(0, 2500),
        )
        preview = self.sample_buffer(opaque, requested)

        color = graph.create_texture("color", camera_target=True)
        with graph.add_pass("CopyToCamera") as render_pass:
            render_pass.set_texture("_SourceTex", preview)
            render_pass.write_color(color)
            render_pass.fullscreen_quad("Fullscreen Blit")

        camera_result = self.write_buffer(
            opaque,
            "color",
            color,
            source="camera",
        )

        with graph.pass_result(camera_result):
            graph.screen_ui_section(resources={"color", "depth"})

        with graph.add_present_pass("Present") as present_pass:
            present_pass.present(color)
```

`present(color)` 是带类型的终止 Action，同时会调用 `set_output(color)`。Graph 也可以只使用 `set_output()`，省略 Present Pass；本例显式展示 Camera Target 与导出边界。没有显式输出时，`graph.build()` 会选取第一张 Camera Target，生产管线仍应明确表达目标输出。

验证时应看到 Camera 输出未受光照的 Base Color，RenderStack 拓扑包含标准帧尾与 **Present**，Console 中没有 Graph 校验错误。诊断期间可以在 `define_topology()` 最后一行暂时加入 `print(graph.get_debug_string())`，记录资源、Action、Access 与输出；检查完成后删除该输出。

逻辑名称是 Graph 范围内的资源身份。可复用片段可以使用 `graph.name_scope()`，避免生成的名称冲突。Pass 顺序按书写顺序记录；声明的 Access 为 Native Compiler 提供依赖与转换信息。Native Compiler 可以剔除无法到达输出且未声明 Side Effect 的工作，因此编写顺序本身不能证明执行。

`screen_ui_section()` 会把 Camera UI、后处理挂点、显示编码和 Screen UI 放在调用位置。通过 RenderStack 构建 Graph 时，缺失的标准后处理挂点和 Screen UI 帧尾也会由安全网补齐。显式调用可以清楚控制它们的位置。单独使用 `RenderPipeline.render()` 时没有这层 RenderStack 安全网，管线需要自行完成输出契约。

标准帧尾使用逻辑资源 `color`。特殊 RenderStack 管线可以把另一张纹理设为最终输出，同时仍需为标准 UI 与显示编码保留有效的 `color` 路径。

## Provider 契约 {#providers_1}

Geometry Buffer Provider 是使用 `@geometry_buffer` 注册在管线类上的方法。注册键为 `(semantic, phase)`，默认 Phase 是 `opaque`。依赖项使用语义名，编译器按拓扑顺序安排 Provider。

```python
import infernux as inx


class ObjectIndexPipeline(inx.renderstack.RenderPipeline):
    name = "Object Index"

    @inx.renderstack.geometry_buffer("object_index", dependencies={"depth"})
    def provide_object_index(self, context):
        target = context.graph.create_texture(
            f"{context.source}_object_index",
            format=inx.rendergraph.Format.RG32_UINT,
        )
        with context.graph.add_pass(
            f"{context.source}_object_index"
        ) as render_pass:
            render_pass.read(context.sample("depth"))
            render_pass.write_color(target)
            render_pass.draw_renderers(
                queue_range=context.queue_range,
                sort_mode=context.sort_mode,
                material_pass="picking",
            )
        return target
```

Provider 接收 `GeometryBufferProviderContext`。它可以读取 `context.graph`、`context.source`、`context.phase`、`context.queue_range`、`context.msaa_samples`、`context.sort_mode`、`context.clear`，并通过 `context.sample()` 取得已有 Semantic Texture。Provider 必须返回非空的 Graph `TextureHandle`；`geometry_stage()` 会以 Decorator 中的 Semantic 发布该 Handle。Dependency 表示同一 Phase 中必须已存在或可由 Provider 生成的其它 Semantic。

派生类声明相同的 Semantic 与 Phase，即可替换内置 Provider。同一个类里为同一注册键声明两个 Provider 会产生歧义并被拒绝。依赖缺失或形成环时，拓扑构建也会失败，错误中会带 Source 与依赖链。每次拓扑构建只会在 Semantic 被请求时运行相关 Provider；API 没有定义跨 Graph 的 Provider 实例缓存。构建局部 Handle 应留在 Context 中，持久 CPU 策略可以保存在 Pipeline 实例上。

在 `define_topology()` 中，应先调用 `self.require_buffer("object_index")`，再进入对应的 `geometry_stage()`。Stage 从传入的 Buffer 集合开始，只运行当前需求涉及的 Provider，最后返回 `PassResult`。RenderStack 中已挂载 Effect 声明的 Geometry 需求会在管线构建前加入 Graph，因此未使用的内置 Normal、Motion 等 Provider 不会生成资源。

```python
requested = self.require_buffer("object_index")
result = self.geometry_stage(
    graph,
    "opaque",
    buffers={"color": color, "depth": depth},
    queue_range=(0, 2500),
)
object_index = self.sample_buffer(result, requested)
```

`require_buffer()` 只在 RenderStack 或基础 DSL 实现已设置 Defining Graph 时有效。它返回的是 `Infernux.renderstack` 中的语义请求 `BufferHandle`；`graph.create_buffer()` 返回的是瞬态 GPU Buffer Handle，两者类型职责不同。Standalone Override 应先调用 `graph.require_geometry_buffers({"object_index"})`，再把同一 Graph 传给 `geometry_stage()`。

## PassResult、Handle 生命周期与 Native Action {#pass-results_1}

`PassResult` 是带 Source 作用域的语义映射，把 `color`、`depth`、`normal`、`motion` 等名称指向 Texture Handle。`source` 在一次 Graph 构建中必须唯一。每次发布或派生 Result 时，Graph 都会分配递增的 `revision`。

```python
before = self.publish_result(
    "opaque",
    {"color": color, "depth": depth},
)

copied = graph.create_texture(
    "post_color",
    format=inx.rendergraph.Format.RGBA16_SFLOAT,
)
with graph.add_pass("CopyColor") as render_pass:
    render_pass.set_texture("_SourceTex", before.sample("color"))
    render_pass.write_color(copied)
    render_pass.fullscreen_quad("Fullscreen Blit")

after = self.write_buffer(
    before,
    "color",
    copied,
    source="post_copy",
)
```

`write_buffer()` 会派生一份替换了单个 Semantic 的 Result。Parent 仍指向较早的纹理，后续拓扑可以有意采样任一 Revision。Revision 只在当前 Graph 构建内表示发布顺序，不代表帧号、持久资产 ID 或可变 GPU 资源版本。

惰性 Geometry Provider 可以把缺失的 Semantic 加入拥有它的 Result。一次写入派生出新 Result 后，早期 Result 的语义绑定仍保持原值。

`publish_pass_result()` 只接受由 Semantic 名称映射到 Graph `TextureHandle` 的数据，同一次 Graph 构建中的 `source` 必须唯一。`PassResult.sample()` 返回逻辑 Handle；`snapshot` 返回当前 Semantic 映射的只读副本。发布 Result 不会分配、复制或修改 GPU Image。仍需由 Pass 声明写入纹理，并由下游 Pass 声明读取。

Texture Handle 与 GPU Buffer Handle 是一次 Builder 运行所拥有的轻量逻辑名称记录。不要把它们保存在 Pipeline 实例上，不要在重建后继续使用，也不要传给另一个 Graph。Python Handle 类型不携带 Graph ID，因此同名的跨 Graph 错误可能绕过早期身份检查。最终的 `RenderGraphDescription` 保存名称与资源描述，实际资源由每相机 Native Graph 创建。

同一个 Graph 中的所有 Camera Target 都指向相机的物理颜色输出，声明多张时会产生警告。持久的 `inx.RenderTexture` 则拥有独立的资源生命周期：可以赋给 `Camera.target_texture`、`UIImage.texture` 或材质纹理参数，也可以通过 `graph.import_texture()` 显式接入渲染图。重建图不会销毁该资源。这些运行时引用不生成资产 GUID，也不会自动保存成纹理资产。

尺寸可以指定为固定像素，也可以相对于 Game 的渲染分辨率。资源在引擎线程创建，例如放在组件的 `start()` 中：

```python
PixelFormat = inx.rendergraph.Format

fixed = inx.RenderTexture(641, 401, depth_format=PixelFormat.D32_SFLOAT)
half_size = inx.RenderTexture(
    scale=(0.5, 0.5),
    format=PixelFormat.RGBA16_SFLOAT,
    depth_format=PixelFormat.D32_SFLOAT,
    samples=4,
)
camera.target_texture = half_size
image.texture = half_size
```

这里的基准是 **Game 实际渲染像素**，不是 Game 面板的显示缩放、桌面 DPI、Scene View 的尺寸或最后绘制的相机尺寸。Game 为 641×401 时，`half_size.width, half_size.height` 为 `(321, 201)`，不足一个像素的部分向上取整。两个比例必须为正的有限数值，大于 1 表示超采样。Game 分辨率变化时，同一资源及其 Camera、材质、UI 引用一起更新，不需要逐帧调用 `resize()`。

固定尺寸资源的 `scale` 为 `None`，可以用 `resize(width, height)` 修改像素尺寸；相对尺寸资源的 `scale` 只读，不能再用像素 `resize()`。修改尺寸模式、比例、格式、深度或 MSAA 时，创建新资源并替换引用。分配失败会保留旧分配并报告错误，不静默降低尺寸或替换格式。相机管线使用 TAA 等需要读取深度的效果时，应显式设置 `sampled_depth=True`；未声明这个用途的深度采样会在图提交前被拒绝。`storage=True` 声明 storage image 用途，普通光栅绘制不需要它。

参考尺寸的概念可对照 [Unity RTHandle 的比例分配](https://docs.unity.cn/Packages/com.unity.render-pipelines.core%4016.0/manual/rthandle-system-using.html)，但 Infernux 对外仍然只有 `RenderTexture`。这里按当前分辨率分配向上取整后的尺寸，也会随分辨率缩小，不承诺 Unity 按最大参考尺寸保留分配的策略。

需要复用资产时，在 Project 中选择 **创建 > 渲染纹理**。新建资产默认带 D32 深度，可直接交给 Camera；只需要颜色的用途可以在 Inspector 中关闭深度。Inspector 编辑的仍是同一份尺寸、格式、MSAA 和用途描述，支持普通的自动保存、撤销和重做。只选中资产不会分配 GPU 纹理。导入器生成 `Library/Artifacts/RenderTexture/<guid>.inxrtex`，Cook 收集的是这份二进制描述，不是源 JSON，也不是渲染出来的像素。

在脚本中声明普通序列化字段，再到 Inspector 中挂上资产即可：

```python
import infernux as inx

class Monitor(inx.InxComponent):
    output: inx.RenderTexture

    def start(self):
        self.game_object.get_component(inx.Camera).target_texture = self.output
```

供 Camera 使用的目标需要配置深度格式。`inx.RenderTexture.load("Assets/Rendering/Monitor.rendertexture")`、`load_by_guid(guid)` 和序列化引用解析到同一份 GPU 资源。重导入会更新已有使用者，不要求脚本重新赋值；Camera 仍持有目标时，不允许移除其深度附件。固定/相对尺寸模式在资产上编辑，运行时 `resize()` 不会写回源文件。只有导入的目标具有 `guid` 和 `file_path`，运行时创建的目标仍为瞬态资源。

也可以直接把资产拖到 **Camera → 目标纹理**，不需要写脚本。这个槽位随场景保存，在编辑和运行模式下均有效。删除资产会保留缺失引用，不会把这台相机改成屏幕输出；恢复同一个 GUID 后，输出会重新连接。Player 解码场景时不读取源文件：Camera 保留 GUID，由渲染器准备导入后的二进制描述。

材质的纹理槽也支持导入的 RenderTexture。可以从 Project 拖入，或调用 `material.set_texture("texSampler", target)`：导入资源保存 GUID，临时创建的资源只覆盖运行时绑定。重导入会更新现有消费者，包括运行时材质副本；删除资源会断开绑定，但保留已保存的 GUID，恢复资产后会重新连接。

把这份材质交给 MeshRenderer 或 `UIImage.material`，就能显示相机画面，不需要先把图像读回 CPU。这条 Project → Camera → Material 流程与 [Unity 的 Render Texture 示例](https://docs.unity3d.com/6000.0/Documentation/Manual/output-to-render-texture.html) 对齐。

不需要自定义材质时，可以直接把 Texture 或 RenderTexture 资产拖入 **UIImage → Texture**。这一个槽位随场景保存具体资产类型和 GUID，不保存渲染出的像素；脚本赋值 `image.texture = imported_target` 也使用同一份引用。赋入临时创建的 RenderTexture 时，仅覆盖运行时画面，赋入 `None` 可撤掉该覆盖。Inspector 中的清空则通过场景 Undo 修改保存的资产槽。旧场景的 `texture_path` 在读取时转换，之后不再写入。Windows Player 从 `Content.inxpkg` 读取 GUID 对应的编译描述，不依赖原始 `.rendertexture` 文件；Camera、材质和图像控件同时取样的流程已经过独立 Windows 包验证。其它平台仍需单独验收。

Camera 矩阵和材质／绘制矩阵入口统一使用 NumPy `(4, 4)` 数组，按 `[行, 列]` 索引，不需要手动展平或转置。例如，平面镜着色器声明 `Mat4 reflectionVP` 后，可以把反射相机的矩阵直接设为该 Renderer 的参数：

```python
view_projection = reflection_camera.projection_matrix @ reflection_camera.view_matrix
mirror_renderer.set_parameter("reflectionVP", view_projection)
```

修改材质本身用 `Material.set_matrix(name, matrix)`，显式绘制用 `DrawParameterBlock.set_matrix(name, matrix)`。Renderer 覆盖不会改动共享材质；这些入口复制传入的数值，之后修改原 NumPy 数组不会自动更新材质。已有的16元素平铺输入和保存的矩阵文档仍按列主序解释。传递矩阵本身不会改变相机的裁剪空间约定。

`graph.create_temporal_history(name, format=..., size=... / size_divisor=...)` 创建每个 View 独立的单采样历史读写对。创建或失效后的首次读取为零，每次执行都需写入输出；引擎负责双缓冲交换和资源退休。分配历史不自动开启相机抖动，TAA 通过 `graph.set_temporal_jitter()` 单独请求。不要写入上一帧的输入，也不要在同一个 Pass 读写同一张图像。

发生小幅瞬移，或其它自动相机跳变检测无法识别的不连续变化时，调用 `camera.reset_history()`。使用这台 Camera 的各个 View 会在下次渲染前重置已有历史资源及上一帧相机矩阵，不影响其它 Camera，也不修改 Transform、投影或场景文件，不需要重建图。同一帧前的多次调用只造成每个 View 重置一次，随后正常积累历史。它与 [URP 的 resetHistory](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@17.0/api/UnityEngine.Rendering.Universal.UniversalAdditionalCameraData.html#UnityEngine_Rendering_Universal_UniversalAdditionalCameraData_resetHistory) 职责相近，但不要求再挂一个额外的 Camera 组件。

Pass Builder 记录一个带类型的 Action，例如 `draw_renderers()`、`fullscreen_quad()`、`copy_texture()` 或 `present()`。再次调用 Action 方法会替换该 Pass 的前一项 Action。Python 不会收到 Native Pass Callback、Command Encoder、Vulkan Handle 或已解析的 GPU 资源。`graph.build()` 把声明序列化为 `GraphPassDesc` 与 `GraphCommandDesc`；`context.apply_graph()` 再把描述交给 Native Compiler 与 Executor。

## 资源 Usage 与 MSAA {#resource-usage_1}

当前 Python Texture API 从 Pass 声明中推导 Usage：

- `write_color()` 与 `write_depth()` 声明 Attachment 写入。
- `read()` 声明 Texture 依赖；`set_texture()` 还会同时记录 Shader 绑定。
- `write_resolve()` 声明 Color Resolve 目标。
- `copy_texture()` 在 Copy Pass 中声明 Transfer Source 与 Destination Access。

因此，`create_texture()` 没有公开的 `usage=` 参数。每项实际用途仍需对应的 Pass 声明。需要采样的 Depth Texture 必须出现在 Read 或 Sampler Binding 中；只有 Attachment 声明时，Graph 看不到采样依赖。

瞬态 GPU Buffer 在创建时显式声明标志：

```python
draw_data = graph.create_buffer(
    "draw_data",
    64 * 1024,
    storage=True,
    indirect=True,
    transfer_destination=True,
)
```

`read_buffer()` 与 `write_buffer()` 会根据这些标志校验 Storage、Indirect 或 Transfer Access。`copy_buffer()` 会给两端 Handle 补充 Transfer Source/Destination 标志。这些声明负责描述访问与同步；资源还需要被可执行 Pass Action 实际使用。

`samples = graph.set_msaa_samples(1|2|4|8)` 声明屏幕管线的采样偏好，并返回实际采样数。绑定了 `Camera.target_texture` 时，以目标资源的采样数为准。创建多采样附件和 Resolve Pass 必须使用返回值，不要写死采样数，也不要为不同相机反复改写作者参数。RenderStack 和独立管线按这个采样配置复用不同的图。`set_msaa_samples(0)` 把屏幕采样设置交给原生层，不适合用来决定显式 Resolve 拓扑。Camera Target 和场景尺寸的 Depth Texture 默认继承 `samples=0`，其它瞬态 Texture 默认单采样；同一 Raster Pass 的颜色与深度附件必须使用相同采样数。

多采样 Color 需要变成单采样 Texture 时，使用 `write_resolve()`：

```python
samples = graph.set_msaa_samples(4)
resolved = graph.create_texture(
    "route_color", format=inx.rendergraph.Format.RGBA16_SFLOAT, samples=1,
)
color = resolved
if samples > 1:
    color = graph.create_texture(
        "route_msaa", format=inx.rendergraph.Format.RGBA16_SFLOAT, samples=samples,
    )
depth = graph.create_texture(
    "depth", format=inx.rendergraph.Format.D32_SFLOAT, samples=samples,
)

with graph.add_pass("Route") as render_pass:
    render_pass.write_color(color)
    render_pass.write_depth(depth)
    if samples > 1:
        render_pass.write_resolve(resolved)
    render_pass.draw_renderers(queue_range=(0, 2500))
graph.set_output(resolved)
```

该 Pass 必须只有 Slot `0` 这一项 Color 输出。Source 必须为多采样；Target 必须是格式与尺寸匹配的瞬态单采样 Color Texture。当前 Python API 没有 Depth Resolve 操作。

## 绘制指定的 Renderer {#renderer-selection}

描边、对象分组遮罩等效果可以用 `RendererSelection` 选择普通 MeshRenderer 和 submesh，而不必复制网格或修改它们的原材质。以下对象应在管线初始化时创建一次；`mask_material` 是项目材质，`target_renderer` 是 Inspector 引用或游戏逻辑选中的 Renderer：

```python
import infernux as inx

selection = inx.rendergraph.RendererSelection(mask_material)
parameters = inx.rendergraph.DrawParameterBlock()
parameters.set_color("baseColor", (1.0, 0.2, 0.05, 1.0))
selection.set(target_renderer, parameters=parameters)

with graph.add_pass("SelectedMask") as p:
    p.write_color(mask)
    p.write_depth(mask_depth)
    p.set_clear(color=(0, 0, 0, 0), depth=1.0)
    p.draw_renderers(renderer_selection=selection)
```

`mask` 和 `mask_depth` 是该图已创建的颜色/深度资源，尺寸和采样数必须匹配。遮罩使用独立深度还是场景深度，由项目的效果决定。颜色、对象 ID 等参数必须由选择材质的 ShaderInfo 声明；不会写入其它对象的共享材质，也不会按同名字段跨参数域查找。

`selection.set(renderer, submesh=2, parameters=parameters)` 只选择一个 submesh；默认 `submesh=-1` 选择全部。两种条目同时存在时，精确 submesh 条目优先。每次 `set()` 都捕获参数值，随后修改 `parameters` 不会追溯修改已记录的条目。`remove(renderer, submesh=2)` 删除精确条目；`clear()` 清空集合。

在 Update/LateUpdate 等引擎更新阶段修改同一个集合即可，不必重建图。图持有集合及其材质，原生绘制使用相机当帧的几何、变换、蒙皮和 GPU 顶点缓冲。集合与相机可见性、原材质 Queue/Pass Tag 过滤取交集；不会将粒子或其它未选对象混入遮罩。被删除或替换的 Renderer 不会因对象编号复用而重新命中。选择材质替换本次绘制的着色程序，因此原材质自定义顶点着色器中的变形仍需在选择材质中实现；这与共享 GPU 网格已经完成的变形不同。

这一接口提供了与 Unity [`CommandBuffer.DrawRenderer`](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Rendering.CommandBuffer.DrawRenderer.html) 类似的 Renderer/submesh/替换材质职责，但将集合接入现有 RenderGraph，不引入另一条渲染调度路径。逐 draw 参数沿用 `DrawParameterBlock`；其中纹理覆盖当前限于普通纹理 GUID，RenderTexture 输入放在选择材质或图的显式输入上。

## 对象数据遮罩 {#object-data-mask}

替换材质不一定输出表面颜色，也可以输出项目需要的数据。例如，描边效果可以存储逻辑对象 ID、正值 eye depth、非法状态，以及斜线和遮挡标记。通过 `RendererSelection` 登记对象后，引擎仍绘制它们的当帧几何。同一 Rigidbody 下的零件如何分组由项目决定，不把编辑器 picking ID 当成游戏逻辑的分组 ID。

创建名为 `ObjectDataMask.frag` 的着色器资产，在 Inspector 中将它赋给材质，并搭配内置 `Standard` 顶点着色器：

```glsl
#version 450
ShaderInfo {
    Name "ObjectDataMask"
    ShadingModel Unlit
    CastShadows Off
    Capabilities [ForwardOnly, NoDepthPass, NoPicking, NoMotionVectors, NoNormalPass, NoBaseColorPass]
    Properties {
        Float ownerId = 1.0
        Float invalid = 0.0
        Float flags = 0.0
    }
}
void main() {
    outColor = vec4(material.ownerId, v_ViewDepth, material.invalid, material.flags);
}
```

这里显式关闭辅助变体，是因为这个 `main()` 输出项目的数据格式，并不输出引擎规定的阴影、拾取、运动、法线或反照率格式。`NoDepthPass` 只关闭自动生成的 depth-only 变体，**不会**关闭遮罩绘制本身的深度测试和写入。通过 MCP 编写完整材质文档时，要同时包含着色器引用、声明的属性及默认值；仅修改着色器名称，不等于执行 Inspector 的着色器赋值操作。

数据图使用单采样 `RGBA32_SFLOAT`，搭配相同尺寸的独立 `D32_SFLOAT` 深度图，例如 `owner_mask_depth`。零表示清空背景，连续分组 ID 上限设为 `2**24 - 1`；半精度浮点不能在这个规模下保持整数 ID 精确。用 `texelFetch` 读取 ID 和标记，不对它们做颜色滤波或 MSAA 平均。本例可用 `int(hatched) + 2 * int(depth_tested)` 编码两个标记。通过 `DrawParameterBlock` 设置浮点值，再登记每个 Renderer；参数按 draw 捕获。

在不透明物体之后，用独立深度绘制遮罩，再读取复制的场景深度，将描边合成到场景颜色。这样既保留墙后选中物体的数据，也能处理选中集合内部的遮挡。替换材质不会自动继承原材质的透明裁剪或顶点着色器变形；需要这些表现时，应在替换着色器中实现。已经写入 GPU 网格缓冲的变形则直接使用当帧几何。

对裁剪表面，在两个着色器中声明覆盖参数，并将源材质的值传给对应的选中 Draw。表面和遮罩使用相同的 UV 判定及面剔除规则。切换为实心材质时，显式恢复遮罩的实心覆盖参数，避免复用参数块时留下前一个 Draw 的裁剪值。这是项目编写的材质合同，不是自动翻译任意着色器。

若项目需要“被遮挡时显示斜线”，可以将对象深度与一个小十字邻域内最远的场景深度比较，抑制细小遮挡物造成的斜线噪点，同时保留宽墙后的斜线。采样半径按像素计算，绝对／相对深度偏移由项目配置；半径设为零可对照逐像素判定。这个容差只影响自动斜线，不应用来绕过显式开启的描边深度测试。普通透明混合通常不写不透明阶段的深度；透明裁剪遮挡物则必须先丢弃孔洞像素，再写深度。选中玻璃时描绘几何外轮廓，与选中叶片时描绘裁剪后轮廓，是两种不同的遮罩策略，应由项目明确选择。

## 在不同阶段绘制世界 UI {#world-ui-stages}

`draw_world_ui(layer_mask=...)` 按 GameObject 的层选择世界 UI。掩码是无符号32位整数，还会与当前 Camera 的可见层取交集；`0` 表示不绘制，默认 `0xffffffff` 包含相机可见的所有层。它承担 Unity [Render Objects](https://docs.unity3d.com/6000.0/Documentation/Manual/urp/renderer-features/renderer-feature-render-objects.html) 中“在哪个阶段绘制哪些层”的职责，不需要另一套 UI 渲染器或相机。

如果希望特定文字不受后处理影响，先从普通世界 UI 中排除它们，再晚一点绘制：

```python
labels = 1 << 30  # 将这些文字的 GameObject 设为第30层。
graph.camera_ui_section(world_ui_layer_mask=0xffffffff ^ labels)

# 在这里加入项目的后处理 Pass。

with graph.add_pass("LateLabels") as p:
    p.write_color("color").write_depth("depth")
    p.draw_world_ui(layer_mask=labels)
graph.screen_ui_overlay_section()  # 显示编码及屏幕叠加 UI。
```

世界 UI 仍按正常规则测试该 Pass 声明的深度；晚绘制不等于穿墙显示。项目也可以选择较早阶段的兼容深度快照，但必须显式生成快照。两组掩码应互不重叠，避免同一个元素画两次。`screen_ui_section(world_ui_layer_mask=...)` 会把同样的过滤条件传给普通世界 UI Pass，预先声明的 Pass 则保留原设置。屏幕空间 UI 不受影响。RenderTexture 依赖也只从该 Pass 实际绘制的元素收集。

## 全屏深度与混合 {#fullscreen-raster-state}

这些控制分别承担 Unity 中 [ZTest](https://docs.unity3d.com/6000.0/Documentation/Manual/SL-ZTest.html)、[ZWrite](https://docs.unity3d.com/6000.0/Documentation/Manual/SL-ZWrite.html) 和常规透明 [Blend](https://docs.unity3d.com/6000.0/Documentation/Manual/SL-Blend.html) 的职责，在这里通过图的 Pass 声明；并不代表已覆盖 ShaderLab 的所有渲染状态。

全屏效果默认替换颜色，不测试或写入深度。需要遮挡或叠加的效果可以显式声明状态，使用同一个 RenderGraph 和渲染设备：

```python
import infernux as inx

# scene_color / scene_depth 已由本帧的场景 Pass 写入。
# previous_depth 与 scene_depth 的格式、尺寸和采样数相同。
graph.add_copy_pass("PreserveSceneDepth").copy_texture(scene_depth, previous_depth)
with graph.add_pass("DepthAwareOverlay") as p:
    p.write_color(scene_color).write_depth(scene_depth)
    p.set_texture("sceneDepthTex", previous_depth)
    p.fullscreen_quad(
        "MyDepthAwareOverlay",
        depth_test=inx.rendergraph.DepthCompare.ALWAYS,
        depth_write=True,
        alpha_blend=True,
    )
```

`depth_test` 接受 `DepthCompare`，例如 `LESS_EQUAL` 或 `ALWAYS`；`None` 关闭测试。`depth_write=True` 必须同时启用测试，无条件写入用 `ALWAYS`。Fragment Shader 可写 `gl_FragDepth`，数值使用当前 Vulkan 视图的设备深度（0—1），而非线性 eye depth；若改变像素深度，应在每个未 discard 的路径上赋值。

`alpha_blend=True` 使用非预乘的 source-over 混合：颜色为 `src.rgb * src.a + dst.rgb * (1-src.a)`，alpha 为 `src.a + dst.a * (1-src.a)`。不要先把背景混入输出颜色再开启混合，否则背景会重复计算。此时颜色附件会保留上一 Pass 的内容；深度测试失败的像素同样保留原色。新建瞬态附件必须通过 `set_clear()` 或此前的写入初始化，不能加载未定义内容。

同一 Pass 不能一边采样深度纹理一边将其作为深度附件；需要读取合成前深度时，像上例一样显式复制，保持格式、尺寸和采样数一致。独立深度附件、只测试不写入和片段深度写入均可组合，不需要引入效果专用的引擎类。

全屏 Shader 在 `Resources` 中声明 `Texture2DMS` 或 `Texture2DMSUInt`，即可用 `texelFetch(texture, pixel, sampleIndex)` 读取多采样颜色或深度的原始样本。物体 ID 和标记应逐样本保留，不能求平均。普通 `Texture2D` 输入则接收解析后的颜色或深度。引擎在构图时从 Shader 反射这一合同；热更新 Shader 后，对应的解析步骤也会重新生成。不能把单采样纹理绑定到 `Texture2DMS`。

需要逐样本合成时，用 `gl_SampleID`、`gl_SamplePosition` 处理各采样点，写入采样数匹配的颜色附件，最后解析颜色；设备需支持 sample-rate shading。只给场景颜色开启 MSAA，不会自动平滑后来由单采样 Shader 生成的描边。`graph.set_msaa_samples(4)` 返回本 Camera 实际使用的采样数，固定 RenderTexture 目标可能覆盖请求值；据此选择单采样或多采样入口，两者可通过 ShaderInfo `Imports` 共用效果代码。非全屏材质消费 MSAA 颜色时，仍需使用 `write_resolve()` 等产生的已解析纹理。

根管线中与视口同尺寸的 `depth` 是约定的 Camera 深度附件。其它名称的深度纹理即使尺寸相同，也各自拥有独立图像；对象遮罩深度和深度快照应使用不同名称。固定尺寸、降采样及导入的 RenderTexture 深度保持自己的资源身份。

## 当前能力边界 {#current-boundaries_1}

公开的 Python RenderGraph 当前提供三种 Pass：

- `add_pass()` 创建 Raster Pass，用于 Renderer Draw、天空、Screen UI 或 Fullscreen 工作。
- `add_copy_pass()` 创建 Copy Pass，执行 `copy_texture()` 或 `copy_buffer()`。
- `add_present_pass()` 通过 `present()` 导出 Color Texture。

Compute Dispatch 尚未进入当前 Python API。现有接口没有 `add_compute_pass()`、`dispatch()` 或 Python `GraphPassType.COMPUTE`。Storage 与 Indirect Buffer Usage 标志可用于资源契约，但不会生成 Dispatch 或 Indirect Draw 命令。

可以按工作负载选择当前可用路径：

- 图像空间计算适合 Fragment 工作时，使用带 `fullscreen_quad()` 和显式 Texture Input 的 Raster Pass。
- 普通场景提交使用 `draw_renderers()` 与 Material Queue 过滤；Renderer Batching 与 Draw Submission 由引擎管理。
- GPU 粒子模拟与 GPU-driven 粒子绘制使用 Particle Graph 子系统，其 Compute 与 Indirect 路径由引擎管理。
- 通用 Compute Kernel、GPU 生成的 Indirect Draw、自定义 Queue 或自定义 Native Resource Import 当前不适合 Python RenderGraph。项目管线调用这些能力之前，需要先增加引擎管理的 Native Feature 以及新的公开 Binding/IR 契约。

Transfer 支持范围较窄。Texture Copy 要求两张不同的瞬态 Texture 格式一致，Camera Target 不参与该操作。Buffer Copy 可指定字节数，且不能超过两端较小者。任意 Blit、格式转换、队列选择与自定义 Transfer Command 目前没有公开的 Python Builder 入口。

如果要将后处理结果写回相机，并继续绘制后续 Pass，应使用内置的全屏着色器，而不是 Copy Pass。相机输出到项目中的 RenderTexture 资产时也遵循这一规则：

```python
with graph.add_pass("CommitGrade") as render_pass:
    render_pass.write_color(camera_color)
    render_pass.set_texture("_SourceTex", graded)
    render_pass.fullscreen_quad("Fullscreen Blit")
```

这里的 `graded` 是前序 Pass 产出的独立单采样颜色纹理，`camera_color` 是图中的相机目标。这个 Pass 不能同时采样和写入同一张纹理。如果只是导出最终画面，不再向相机目标追加绘制，则使用 `present()`。

## 校验、诊断与恢复 {#debugging_1}

`graph.build()` 会检查资源与 Pass 重名、资源缺失、Action 与 Pass Type 不匹配、Buffer Usage、Attachment 格式、采样数一致性、Resolve 契约、扩展点位置和最终输出。随后由 Native Compiler 校验并调度生成的 Graph。

拓扑编辑失败时，RenderStack 拒绝这次修改，保留该输出采样配置上最后一次成功的图及其配套效果绑定；Inspector 在保留已有拓扑视图的同时报告错误。如果从未构建成功，Editor 和 Player 都明确报错，不会换成 Default Forward。修复被监听的管线文件或修改管线参数会清除失败状态。这是作者编辑事务的处理，不代表允许读取未写入的纹理，也不代表可以偷偷拿旧相机画面充当本帧输出。

Standalone `RenderPipeline.render()` 没有上一份有效图或 Default Forward 恢复。`define_topology()` 或 `build()` 失败后，`_standalone_desc` 保持为空，异常继续可见，后续 Render 调用会重试。管线被替换时，`dispose()` 会清除旧的 Standalone Description。

当前 Editor 没有图形化 RenderGraph Debugger。可使用 `graph.get_debug_string()` 查看资源、Pass Action、读写、Resolve 与输出的文本摘要，再到 Editor 和真实构建中确认行为。这段文本描述一份 Topology Artifact，无法区分每台相机的 Native 实例。多相机运行日志应同时包含 Host 可取得的 Camera 身份、`context.graph_instance_id` 与 `RenderGraphDescription.source_revision`；Graph Instance ID 用于区分 Native Graph 实例，Source Revision 标识共享的 Python 拓扑。

低层管线交付前，至少检查以下几点：

1. 每个被消费的 Texture 或 Buffer 都有对应 Read/Access 声明和上游 Producer。
2. Attachment 采样数一致，每条会被采样的 MSAA Color 路径都在预期位置完成 Resolve。
3. EffectStage 的 Input/Output Contract 与该 Source 可用的语义资源一致。
4. 每台相机拥有独立的 Graph Runtime 资源，以及相机局部的光照和阴影状态。
5. Graph 只有一个最终 Color 输出、一次显示编码，并明确放置 Camera UI 与 Screen UI。
