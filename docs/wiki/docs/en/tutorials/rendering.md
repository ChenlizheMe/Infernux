# Rendering & Post-Processing — Make It Pretty

<div class="class-info">
Tutorial &nbsp;|&nbsp; <a href="../../zh/tutorials/rendering.html">中文</a>
</div>

## Overview

InfEngine ships with a **Scriptable Render Pipeline** system inspired by Unity URP/HDRP. You can choose a built-in pipeline, tweak its parameters, layer post-processing effects on top, or go full mad-scientist and write your own pipeline from scratch. All from Python. No shader recompilation required (well, for the pipeline part).

The architecture has three layers:

| Layer | What it does | You touch it when… |
|---|---|---|
| **RenderPipeline** | Defines the *topology* — which passes run, in what order | You want a completely custom pipeline |
| **RenderStack** | Manages a pipeline + injectable passes. Lives on a GameObject | You want to add/remove/reorder effects at runtime |
| **FullScreenEffect** | A multi-pass post-processing effect (Bloom, Vignette, etc.) | You want pretty pictures |

## Quick Start: Post-Processing in 60 Seconds

The fastest way to get post-processing running:

```python
from InfEngine import *
from InfEngine.renderstack import *

class SetupPostProcessing(InfComponent):
    def start(self):
        # RenderStack is a component — add it to any GameObject
        stack = self.game_object.add_component(RenderStack)
        # It uses DefaultForwardPipeline automatically
        
        # Add some effects
        bloom = BloomEffect()
        bloom.threshold = 1.0
        bloom.intensity = 0.8
        bloom.scatter = 0.7
        stack.add_pass(bloom)
        
        vignette = VignetteEffect()
        vignette.intensity = 0.4
        stack.add_pass(vignette)
        
        # Tone mapping should be last (HDR → LDR)
        tonemapping = ToneMappingEffect()
        tonemapping.exposure = 1.2
        stack.add_pass(tonemapping)
```

Boom. Your scene now has bloom, vignette, and tone mapping. That was easy.

## Built-In Pipelines

### Default Forward

The workhorse. Renders opaque objects front-to-back, then skybox, then transparent objects back-to-front. Supports shadow maps and MSAA.

```
ShadowCasterPass → OpaquePass → [after_opaque]
→ SkyboxPass → [after_sky]
→ TransparentPass → [after_transparent]
→ [before_post_process] → ScreenUI → [after_post_process]
```

Parameters exposed in the inspector:

| Parameter | Default | Description |
|---|---|---|
| `shadow_resolution` | 4096 | Shadow map size (256–8192) |
| `msaa_samples` | X4 | Anti-aliasing: OFF, X2, X4, X8 |
| `enable_screen_ui` | True | Render screen-space UI |

### Default Deferred

For scenes with many lights. Uses a GBuffer with 4 MRT slots (albedo, normals, material, emission) + depth, then resolves lighting in a fullscreen pass.

```
ShadowCasterPass → GBufferPass (MRT) → [after_gbuffer]
→ DeferredLightingPass → [after_opaque]
→ SkyboxPass → [after_sky]
→ TransparentPass (forward) → [after_transparent]
→ [before_post_process] → ScreenUI → [after_post_process]
```

> **Note:** The deferred pipeline requires a `deferred_lighting` shader. If it's not available, it outputs raw albedo.

### Switching Pipelines

```python
stack = game_object.get_py_component(RenderStack)
stack.set_pipeline("Default Deferred")
# or
stack.set_pipeline("Default Forward")
```

## Post-Processing Effects

All built-in effects inherit from `FullScreenEffect`. They have `serialized_field` parameters that appear in the Inspector.

### Bloom

The classic "bright things glow" effect. Uses a progressive downsample/upsample chain.

```python
bloom = BloomEffect()
bloom.threshold = 1.0      # Minimum brightness to bloom
bloom.intensity = 0.8      # Final bloom strength
bloom.scatter = 0.7        # How far the bloom spreads (0–1)
bloom.clamp = 65472.0      # Max brightness (prevents fireflies)
bloom.tint_r = 1.0         # Tint color (default: white)
bloom.tint_g = 1.0
bloom.tint_b = 1.0
bloom.max_iterations = 5   # Downsample steps (1–8)
stack.add_pass(bloom)
```

**Injection point:** `before_post_process` (order 100)

### Color Adjustments

Exposure, contrast, saturation, and hue shift. The "make everything look cinematic" knob.

```python
ca = ColorAdjustmentsEffect()
ca.post_exposure = 0.5   # EV offset (-5 to 5)
ca.contrast = 10.0       # -100 to 100
ca.saturation = -20.0    # -100 to 100 (negative = desaturated)
ca.hue_shift = 0.0       # -180 to 180 degrees
stack.add_pass(ca)
```

**Injection point:** `before_post_process` (order 200)

### White Balance

Temperature and tint. Makes your scene feel warm and cozy, or cold and dystopian.

```python
wb = WhiteBalanceEffect()
wb.temperature = 15.0   # Positive = warm, negative = cool (-100 to 100)
wb.tint = 0.0           # Green/magenta shift (-100 to 100)
stack.add_pass(wb)
```

**Injection point:** `before_post_process` (order 150)

### Vignette

Darkens screen edges. Instant "cinematic" look.

```python
vig = VignetteEffect()
vig.intensity = 0.35    # 0 = off, 1 = full black edges
vig.smoothness = 0.3    # Falloff softness
vig.roundness = 1.0     # 1 = circular, lower = more rectangular
vig.rounded = False     # Force perfect circle regardless of aspect ratio
stack.add_pass(vig)
```

**Injection point:** `before_post_process` (order 500)

### Chromatic Aberration

RGB channel separation. Subtle = "cinematic lens"; Heavy = "the protagonist just got punched".

```python
chroma = ChromaticAberrationEffect()
chroma.intensity = 0.1   # 0 = off, 1 = strong separation
stack.add_pass(chroma)
```

**Injection point:** `before_post_process` (order 600)

### Film Grain

Cinematic noise. Makes your game look like a movie from the 90s. Or 2020s, because retro is in.

```python
grain = FilmGrainEffect()
grain.intensity = 0.2    # 0 = off, 1 = heavy noise
grain.response = 0.8     # 0 = uniform, 1 = highlights only
stack.add_pass(grain)
```

**Injection point:** `after_post_process` (order 800)

### Sharpen

Contrast Adaptive Sharpening (AMD FidelityFX CAS). Enhances detail without halos.

```python
sharp = SharpenEffect()
sharp.intensity = 0.5    # 0 = off, 1 = maximum
stack.add_pass(sharp)
```

**Injection point:** `after_post_process` (order 850)

### Tone Mapping

Converts HDR to displayable LDR. **Should be the last effect in the HDR chain.**

```python
tm = ToneMappingEffect()
tm.mode = ToneMappingMode.ACES   # ACES, Reinhard, or None_
tm.exposure = 1.0                 # Pre-tonemap exposure multiplier
tm.gamma = 2.2                    # Gamma correction (2.2 = sRGB)
stack.add_pass(tm)
```

**Injection point:** `after_post_process` (order 900)

### Recommended Stack Order

Effects are sorted by their `default_order` within each injection point. Here's the typical order:

```
before_post_process:
  100  Bloom           — extract bright pixels in HDR
  150  White Balance   — adjust color temperature
  200  Color Adj.      — exposure/contrast/saturation
  500  Vignette        — darken edges (still HDR)
  600  Chrom. Aberr.   — lens distortion

after_post_process:
  800  Film Grain      — noise overlay (LDR)
  850  Sharpen         — CAS sharpening
  900  Tone Mapping    — HDR → LDR conversion
```

## The RenderGraph

Under the hood, every pipeline builds a `RenderGraph` — a DAG of texture resources and render passes. You don't need to touch this for basic post-processing, but it's powerful for custom pipelines.

### Textures

```python
from InfEngine.rendergraph import RenderGraph, Format

graph = RenderGraph("MyGraph")

# Camera-sized color target
color = graph.create_texture("color", camera_target=True)

# Depth buffer
depth = graph.create_texture("depth", format=Format.D32_SFLOAT)

# Half-resolution texture (for downsampling)
half_color = graph.create_texture("half_color",
    format=Format.RGBA16_SFLOAT,
    size_divisor=2)

# Fixed-size texture (e.g., shadow map)
shadow = graph.create_texture("shadow_map",
    format=Format.D32_SFLOAT,
    size=(4096, 4096))
```

### Passes

```python
# Context manager style (recommended)
with graph.add_pass("OpaquePass") as p:
    p.write_color("color")
    p.write_depth("depth")
    p.set_clear(color=(0.1, 0.1, 0.1, 1.0), depth=1.0)
    p.set_texture("shadowMap", "shadow_map")
    p.draw_renderers(queue_range=(0, 2500), sort_mode="front_to_back")

# Fullscreen quad pass (for post-processing)
with graph.add_pass("MyEffect") as p:
    p.set_texture("_SourceTex", color)
    p.write_color(output)
    p.fullscreen_quad("my_shader")
```

### Injection Points

Injection points are named slots where user passes can be inserted:

```python
graph.injection_point("after_opaque", resources={"color", "depth"})
```

The `resources` set tells the system which textures are available at that point.

## Writing a Custom Pipeline

Want full control? Subclass `RenderPipeline`:

```python
from InfEngine.renderstack import RenderPipeline
from InfEngine.rendergraph import RenderGraph, Format
from InfEngine.components import serialized_field

class MinimalPipeline(RenderPipeline):
    name = "Minimal Forward"
    
    clear_color_r: float = serialized_field(default=0.1, range=(0, 1))
    clear_color_g: float = serialized_field(default=0.1, range=(0, 1))
    clear_color_b: float = serialized_field(default=0.1, range=(0, 1))
    
    def define_topology(self, graph):
        graph.create_texture("color", camera_target=True)
        graph.create_texture("depth", format=Format.D32_SFLOAT)
        
        cc = (self.clear_color_r, self.clear_color_g, self.clear_color_b, 1.0)
        
        with graph.add_pass("MainPass") as p:
            p.write_color("color")
            p.write_depth("depth")
            p.set_clear(color=cc, depth=1.0)
            p.draw_renderers(queue_range=(0, 5000))
            p.draw_skybox()
        
        # Add injection points so effects can hook in
        graph.injection_point("before_post_process", resources={"color"})
        graph.injection_point("after_post_process", resources={"color"})
        
        graph.set_output("color")
```

Then use it:

```python
stack = game_object.get_py_component(RenderStack)
stack.set_pipeline("Minimal Forward")
```

## Writing a Custom FullScreenEffect

Want to add your own post-processing? Subclass `FullScreenEffect`:

```python
from InfEngine.renderstack import FullScreenEffect
from InfEngine.rendergraph import Format
from InfEngine.components import serialized_field

class InvertColorsEffect(FullScreenEffect):
    name = "Invert Colors"
    injection_point = "before_post_process"
    default_order = 250
    menu_path = "Post-processing/Invert Colors"
    
    strength: float = serialized_field(default=1.0, range=(0, 1))
    
    def get_shader_list(self):
        return ["fullscreen_triangle", "invert_colors"]
    
    def setup_passes(self, graph, bus):
        color_in = bus.get("color")
        if color_in is None:
            return
        
        color_out = self.get_or_create_texture(
            graph, "_invert_out", format=Format.RGBA16_SFLOAT)
        
        with graph.add_pass("Invert_Apply") as p:
            p.set_texture("_SourceTex", color_in)
            p.write_color(color_out)
            p.set_param("strength", self.strength)
            p.fullscreen_quad("invert_colors")
        
        bus.set("color", color_out)
```

Key rules for custom effects:
1. Read from `bus.get("color")` (or whatever resource you need)
2. Create a **new** output texture (don't write back to the input — read/write hazard!)
3. Write your result back to `bus.set("color", output)`
4. The `get_or_create_texture` helper avoids duplicate textures

## Runtime Effect Control

```python
stack = game_object.get_py_component(RenderStack)

# Toggle an effect
stack.set_pass_enabled("Bloom", False)
stack.set_pass_enabled("Bloom", True)

# Remove an effect entirely
stack.remove_pass("Film Grain")

# Change effect order
stack.reorder_pass("Vignette", 50)  # Lower = earlier

# Move a pass before another
stack.move_pass_before("Chromatic Aberration", "Vignette")

# Force graph rebuild after changes
stack.invalidate_graph()
```

## The ResourceBus

The `ResourceBus` is how render passes communicate. It's a string → TextureHandle dictionary:

```python
def setup_passes(self, graph, bus):
    # Read what the previous effect produced
    color = bus.get("color")       # TextureHandle or None
    depth = bus.get("depth")       # might be None if no depth pass wrote it
    
    # ... do your thing ...
    
    # Write back so the next effect gets your output
    bus.set("color", my_output)
```

Available resources depend on the injection point. Typically:
- `before_post_process`: `color`, `depth`
- `after_post_process`: `color` (depth may not be available)
- Custom injection points: whatever the pipeline declares

## Discovering Available Pipelines & Passes

```python
from InfEngine.renderstack import discover_pipelines, discover_passes

# All registered pipeline classes
pipelines = discover_pipelines()
for name, cls in pipelines.items():
    print(f"  {name}: {cls}")

# All registered render pass classes
passes = discover_passes()
for name, cls in passes.items():
    print(f"  {name}: {cls}")
```

## Tips

- **Effect order matters.** Bloom in HDR space looks different from bloom after tone mapping. The built-in `default_order` values give you sensible defaults.
- **Always call `invalidate_graph()`** after modifying the stack at runtime. The graph is cached for performance.
- **Tone mapping should be last** in the post-process chain (or at least after all HDR effects). Otherwise bloom operates on already-compressed values and looks washed out.
- **Read/write hazard**: Never `write_color` to the same texture you're reading from. Always create a new output texture.
- **Performance**: Each effect adds at least one fullscreen pass. On low-end hardware, less is more.

## See Also

- [RenderGraph API](../api/RenderGraph.md)
- [RenderStack API](../api/RenderStack.md)
- [FullScreenEffect API](../api/FullScreenEffect.md)
- [BloomEffect API](../api/BloomEffect.md)
- [ToneMappingEffect API](../api/ToneMappingEffect.md)
- [RenderPipeline API](../api/RenderPipeline.md)
