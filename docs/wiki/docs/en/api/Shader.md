# Shader

<div class="class-info">
class in <b>infernux.core</b>
</div>

## Description

Query, check, and reimport GPU shader assets.

<!-- USER CONTENT START --> description

This page describes the current Slang rendering development branch. `infernux.core.Shader` is a host tooling API; `infernux.shader` supplies declarations and mathematics for static GPU author files.

Author programs in separate `.frag.py`, `.vert.py`, `.shadingmodel.py`, and `.shaderlib.py` assets. The engine analyzes these files statically and uses Slang to produce target programs. They are not imported as component modules, and their functions execute on the GPU.

An asset declares a logical `shader_id`, such as `infernux/lit`. Materials select vertex and fragment assets independently. Selectors use `/` in IDs to build hierarchical menus; a fragment does not choose a vertex asset. Asset GUIDs retain reference identity, while logical IDs support lookup and display.

<!-- USER CONTENT END -->

## Static Methods

| Method | Description |
|------|------|
| `Shader.check(asset: str, *, source: str \| None = None) → dict` | Check an asset or unsaved draft and return its schema and diagnostics for both GPU targets. |
| `Shader.get_schema(asset: str) → dict` | Query logical fields, graph inputs, and the program domain without creating a device pipeline. |
| `Shader.property_to_id(name: str) → int` | Obtain a reusable process-local material property ID. |
| `Shader.is_loaded(name: str, shader_type: str = "vertex") → bool` | Query renderer publication of a stage or linked material program. |
| `Shader.reload(shader_id: str, shader_type: str \| None = None) → bool` | Reimport an Editor shader asset through the asset system. |

<!-- USER CONTENT START --> static_methods

### Checking and publication

`check` accepts an asset path, GUID, or registered unique Shader ID. Its optional `source` argument checks an unsaved draft. The result contains `valid`, `schema`, and `diagnostics`, including source locations. Checking does not write the asset, publish changes, or create a graphics pipeline.

`get_schema` returns the logical schema of the current revision, including `domain`, `info`, `parameters`, and `inputs`. Parameters can declare `range`, `label`, and `hdr` metadata. Inspector uses these declarations directly; a float with `range=(0.0, 1.0)` receives a slider.

`reload` runs in an Editor or headless authoring host. It resolves a path or logical name through the asset reimport flow and raises on failure. For vertex/fragment name lookup, specify `shader_type="vertex"` or `"fragment"` to select a stage. Use a path for an ID ambiguous within one stage. Multiple reimports run sequentially.

`is_loaded` queries publication without loading anything. It returns `False` before startup, after shutdown, or without a renderer. A previously published usable program can remain available after an invalid edit, so this is not a last-edit success flag. Use `check` for drafts; Console tracks actual import errors and resolves them after successful publication.

Player keeps a compiler available to prepare required program combinations or compute signatures. Frozen author assets cannot be modified in place with `reload`. `property_to_id` returns an ID for repeated material updates within one process; persist property names rather than these integers.

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example

Create `Assets/Shaders/warm_surface.frag.py`:

```python
import infernux.shader as shader


class WarmSurface(shader.surface):
    shader_id = "my_game/warm_surface"
    base_color = shader.material.color(default=(1.0, 0.45, 0.1, 1.0))
    smoothness = shader.material.float(default=0.5, range=(0.0, 1.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(
            albedo=self.base_color.rgb,
            smoothness=self.smoothness,
            normal_ws=ctx.normal_ws,
            alpha=self.base_color.a,
        )
```

This is static GPU source. Use engine GPU declarations and supported ShaderLib imports. Keep NumPy, component lifecycle code, and host IO in ordinary Python scripts.

After importing the asset, create a material from an ordinary tool or component:

```python
from infernux.core import Material, Shader

path = "Assets/Shaders/warm_surface.frag.py"
report = Shader.check(path)
print(report["valid"], report["diagnostics"])

material = Material.from_shader(path, "Warm material")
smoothness_id = Shader.property_to_id("smoothness")
material.set_float(smoothness_id, 0.7)
```

The material starts with `infernux/standard`. Inspector can select another compatible vertex asset independently. The engine prepares the surface entries needed for forward rendering, depth, shadows, motion, and related passes.

Set `hidden = True` on an author class to hide it from ordinary selection menus. Its ID and existing references remain usable.

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

[Material](Material.md) · [Vertex stages](../../../../learn/vertex-stage.html) · [Fragments and materials](../../../../learn/fragment-materials.html)

<!-- USER CONTENT END -->
