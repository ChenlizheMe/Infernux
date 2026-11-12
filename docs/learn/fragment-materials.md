<!-- language:en -->

<span class="mini-tag">Custom Rendering · Chapter 3</span>

# Fragment stages, surfaces, and materials

Create a `*.frag.py` GPU asset. Its Python syntax is parsed statically and compiled by Slang; the file is never imported as a game script. A surface describes a material, a shading model evaluates light, and the engine owns geometry passes, target layouts, resources, and publication.

## A first surface {#first-surface}

Save this as `painted.frag.py`:

```python
import infernux.shader as shader


class Painted(shader.surface):
    shader_id = "example/materials/painted"
    base_color = shader.material.color(default=(0.9, 0.2, 0.1, 1.0), label="Color")
    albedo_map = shader.material.texture_2d(default="white")
    smoothness = shader.material.float(default=0.4, range=(0.0, 1.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        texel = shader.sample(self.albedo_map, ctx.uv0)
        return shader.SurfaceData(
            albedo=texel.rgb * self.base_color.rgb * ctx.color,
            alpha=texel.a * self.base_color.a,
            smoothness=self.smoothness,
        )
```

Import the file, assign it to a Material's Fragment field, and assign the Material to a MeshRenderer. The engine supplies the standard instanced/skinned vertex program. In an initialized project you can also create the Material directly:

```python
import infernux as inx

material = inx.Material.from_shader("Assets/Shaders/painted.frag.py", "Painted")
material.set_color("base_color", 0.2, 0.6, 1.0, 1.0)
material.set_float("smoothness", 0.65)
renderer.set_material(material)
```

Project fragment creation now produces a `*.frag.py` surface template. Files use `snake_case`, class names use `CapWords`, and fields and methods use `snake_case`. Imported shader references retain the asset GUID; renaming a display label does not create a second program identity.

`shader_id` is a case-sensitive locator independent of the filename and class name. Built-ins use `infernux/lit`, `infernux/unlit`, and `infernux/post/...`; internal fullscreens use `infernux/internal/...` and `hidden = True`. `inx.Shader.get_schema("infernux/lit")` and `inx.Material.from_shader("infernux/lit")` resolve the registered asset. Ambiguous IDs require a GUID or path instead of choosing an arbitrary file. Without an explicit ID the file stem is the locator.

Other literal class declarations are `hidden`, `surface_type`, `render_queue`, `cull_mode`, `depth_write`, `depth_test`, `blend`, `pass_tag`, `stencil`, `alpha_clip`, `cast_shadows`, and `receive_shadows`. They are asset/default-state controls, not Material uniforms. A transparent surface supplies alpha blend, disabled depth writes, and queue 3000 unless explicitly declared otherwise. `alpha_clip` is a numeric threshold or None. Material state overrides remain effective. Hidden assets still appear in Project; they are excluded from ordinary shader selectors.

## Material values and textures {#properties}

`shader.material` declares float, int, color, vec2/3/4, mat4, and texture_2d properties. `default`, `label`, numeric `range`, and color `hdr` metadata drive the existing Material Inspector. A range supplies a slider; it does not silently clamp shader arithmetic. Boolean, unsigned integer, and mat3 Material ownership are still being extended; fullscreen parameters already support them.

Texture defaults are `"white"`, `"black"`, and `"normal"`. The referenced texture asset owns color space, GPU format, mipmaps, and sampler settings. A shader does not add another sRGB policy. A logical `Color2D` carries its image and sampler; users do not write layouts, descriptor bindings, or register numbers.

The built-in Lit material uses these names:

| Purpose | Properties |
| --- | --- |
| Base color | `base_color`, `albedo_map`, `base_color_uv_set`, `albedo_map_st` |
| Metal | `metallic`, `metallic_map`, `metallic_channels`, `metallic_uv_set` |
| Smoothness / imported roughness | `smoothness`, `smoothness_map`, `smoothness_channels`, `smoothness_from_roughness`, `smoothness_uv_set` |
| Occlusion | `ambient_occlusion`, `occlusion_map`, `occlusion_strength`, `occlusion_uv_set` |
| Normal | `normal_map`, `normal_scale`, `normal_uv_set` |
| Emission | `emission_color`, `emission_map`, `emission_uv_set` |

Each texture has a `<name>_st` vec4: XY tiling and ZW offset. `get/set_texture_scale` and `get/set_texture_offset` use that same property. There are no old-name aliases.

```python
material = inx.Material.create_lit("Stone")
material.set_color("base_color", 0.6, 0.65, 0.7, 1.0)
material.set_texture("albedo_map", texture_guid)
material.set_texture_scale("albedo_map", (2.0, 2.0))
material.set_float("normal_scale", 0.8)
```

Lit selects UV0 or UV1 per map. A normal map on UV0 uses the authored tangent and handedness. UV1 reconstructs its tangent basis from position/UV derivatives; a degenerate UV basis uses the authored tangent. Normal maps reconstruct Z from their RG encoding, including two-channel imported formats.

## Surface and geometry contracts {#surface-contract}

`SurfaceContext` exposes `position_ws`, `normal_ws`, `tangent_ws`, `color`, `uv0`, `uv1`, `view_depth`, object identity, layer, and flags. `SurfaceData` uses linear values:

| Field | Meaning / default |
| --- | --- |
| `albedo` | Base reflectance, vec3(1) |
| `normal_ws` | World normal; zero requests the interpolated geometry normal |
| `metallic`, `smoothness` | Metal fraction 0, smoothness 0.5 |
| `occlusion` | Occlusion multiplier 1 |
| `emission` | Emitted linear RGB, vec3(0) |
| `alpha` | Coverage / opacity 1 |
| `specular_highlights` | Direct/indirect specular multiplier 1 |
| `shading_param0`, `shading_param1` | Two scalar custom-model payloads, 0 |

The same surface evaluation and alpha decision feed visible color, shadow/depth, picking, motion, normal, and base-color passes. PBR and Unlit additionally produce the shared GBuffer with the correct model ID. Previous object transforms and bone offsets feed motion. Render state remains Material state: use its surface/blend, culling, depth, queue, and alpha-clip controls instead of adding a second state block to the author file.

For local deformation, select a `*.vert.py` asset in the Material Inspector's Vertex field, or set `material.vert_shader_name` to its ID. Vertex and Fragment are independent selections; their fields are linked into one program. See [vertex assets](vertex-stage.html).

## Separate shading models {#shading-models}

A surface uses PBR by default. To define a separate model, save `flat.shadingmodel.py`:

```python
import infernux.shader as shader


class Flat(shader.shadingmodel):
    shader_id = "example/shading/flat"
    hidden = True
    lighting_contract = "unlit"

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        return surface.albedo + surface.emission
```

In the surface class, add `shading_model = shader.ref("./flat.shadingmodel.py")` or use its registered ID with `shader.ref("example/shading/flat")`. Alpha remains owned by the surface. `ShadingContext` carries lighting geometry, pixel position, view depth, layer, and shadow reception. It deliberately does not expose a mesh's UV or tangent stream. Models declare `standard_pbr`, `unlit`, or `custom_response` as their lighting contract. An unlit model cannot request lighting functions.

Lighting functions available inside a model are:

```python
color = shader.lighting.pbr(surface, ctx)
ambient = shader.lighting.pbr_indirect(surface, ctx)
light = shader.lighting.main_light(ctx, surface.normal_ws)
direction = shader.lighting.view_direction(ctx)
irradiance = shader.lighting.ambient(ctx, surface.normal_ws)
direct = shader.lighting.direct(surface, ctx)
```

`pbr` includes direct light, the current environment response, and emission. `pbr_indirect` returns the material-weighted environment term only; do not multiply its albedo/BSDF again or add another copy of the same environment. It is not yet the future ray-traced incident-light input. A light's color, attenuation, and shadow are separate fields; main-light access respects the object's layer and light domain.

`ambient` returns environment irradiance before a material response. `direct` traverses the engine's directional, point, spot and rectangle lights and invokes the model's `shade_light` method for each light sample. It preserves separate color, attenuation and visibility values and applies the same light layers, shadow views and Forward+ lists as PBR. It requires this typed method:

```python
def shade_light(self, surface: shader.SurfaceData, ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
    diffuse = shader.max(shader.dot(surface.normal_ws, light.direction), 0.0)
    return surface.albedo * light.color * light.attenuation * light.shadow * diffuse
```

The interface is specialized by Slang at compilation; authors do not copy light loops or add runtime model switches. Call `direct` once from `shade`, then add the desired environment and emission terms. `infernux/toon` uses this path and can be paired with a separately selected vertex asset.

Models may call typed methods and static ShaderLib functions. Surface/model/library dependencies are prepared together and republished on accepted source updates. Built-in PBR, Unlit and Toon participate in Deferred's model dispatch. Custom model assets currently use Forward/Forward+ color evaluation, including the existing Forward route inside a Deferred pipeline; a custom GBuffer/deferred dispatcher is still being implemented.

## Shader libraries and raw fragments {#libraries}

GPU files import `infernux.shader` and explicit static ShaderLib exports. They do not import NumPy, Torch, game components, or arbitrary host modules. For example, `palette.shaderlib.py` can export a typed function and a neighboring shader can use `from .palette import quantize`. Built-in surface helpers are available through `from infernux.shader.lib.surface import texture_uv, sample_normal`. These imports are compile-time dependencies, not Python import execution.

`shader.frag` is a separate raw fragment entry: `main(self, ctx: shader.mesh_input) -> shader.vec4`. It currently supplies Forward/Forward+ color. The engine does not reinterpret that final shaded RGB as an albedo or claim that it automatically supplies surface auxiliary passes. Use `shader.surface` when those shared passes are required. Fullscreen programs use `shader.fullscreen`, graph `shader.input(...)` ports, and `shader.parameter`; they share the same compiler and target-layout service.

## Checking changes {#diagnostics}

`inx.Shader.check(path, source=draft)` checks a file or unsaved draft without publishing a GPU program. It returns validity, logical schema, and source diagnostics for SPIR-V and WGSL. `inx.Shader.get_schema(path)` reads the imported asset's fields. Error locations point to the original author, model, or library file. Host imports and unsupported type/stage combinations are preparation errors.

`inx.Shader.reload(path)` is an authoring operation. Successful changes publish the model/library dependents together; invalid candidates report their source error and do not replace a live valid GPU publication. Normal Material value/texture changes update their bindings without rebuilding the shader or render graph. Test the actual scene, shadows and auxiliary passes as well as the Material preview.

<!-- language:zh -->

<span class="mini-tag">自定义渲染 · 第 3 章</span>

# 片元阶段、Surface 与材质

新建 `*.frag.py` GPU 资产。引擎静态解析它的 Python 语法，再交给 Slang 编译；它不会作为游戏脚本导入执行。Surface 描述材质，ShadingModel 计算光照响应，引擎管理几何通道、目标布局、资源和发布。

## 第一份 Surface {#first-surface_1}

保存为 `painted.frag.py`：

```python
import infernux.shader as shader


class Painted(shader.surface):
    shader_id = "example/materials/painted"
    base_color = shader.material.color(default=(0.9, 0.2, 0.1, 1.0), label="颜色")
    albedo_map = shader.material.texture_2d(default="white")
    smoothness = shader.material.float(default=0.4, range=(0.0, 1.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        texel = shader.sample(self.albedo_map, ctx.uv0)
        return shader.SurfaceData(
            albedo=texel.rgb * self.base_color.rgb * ctx.color,
            alpha=texel.a * self.base_color.a,
            smoothness=self.smoothness,
        )
```

导入文件，赋给 Material 的 Fragment 字段，再将材质赋给 MeshRenderer。引擎提供支持实例和蒙皮的标准顶点程序。在已初始化的工程中，也可以直接创建：

```python
import infernux as inx

material = inx.Material.from_shader("Assets/Shaders/painted.frag.py", "绘制材质")
material.set_color("base_color", 0.2, 0.6, 1.0, 1.0)
material.set_float("smoothness", 0.65)
renderer.set_material(material)
```

Project 新建片元资产现在生成 `*.frag.py` Surface 模板。文件名使用 `snake_case`，类名使用 `CapWords`，字段和方法使用 `snake_case`。导入后的 Shader 引用保留资产 GUID，显示名称改变不会产生第二份程序身份。

`shader_id` 是区分大小写、独立于文件名和类名的定位标识。内置入口使用 `infernux/lit`、`infernux/unlit` 和 `infernux/post/...`；内部全屏程序使用 `infernux/internal/...`，并声明 `hidden = True`。`inx.Shader.get_schema("infernux/lit")` 和 `inx.Material.from_shader("infernux/lit")` 可直接定位资产。重复 ID 要求使用 GUID 或路径，不随机选一个文件。未显式声明 ID 时，文件主干名就是其定位标识。

其它静态类声明包括 `hidden`、`surface_type`、`render_queue`、`cull_mode`、`depth_write`、`depth_test`、`blend`、`pass_tag`、`stencil`、`alpha_clip`、`cast_shadows`、`receive_shadows`。这些是资产及默认状态，不是材质 Uniform。透明 Surface 未明确另设时采用 alpha 混合、不写深度和 3000 队列。`alpha_clip` 为数值阈值或 None。材质显式状态覆盖仍然有效。Hidden 资产仍显示在 Project，只在普通 Shader 选择器中隐藏。

## 材质值与纹理 {#properties_1}

`shader.material` 声明 float、int、color、vec2/3/4、mat4 和 texture_2d。`default`、`label`、数值 `range`、颜色 `hdr` 直接进入已有 Material Inspector。`range` 提供滑条，不会偷偷限幅 Shader 的计算。Material 的 bool、uint 和 mat3 所有者仍在扩展；全屏参数已经支持这些类型。

纹理默认值为 `"white"`、`"black"`、`"normal"`。颜色空间、GPU 格式、mipmap 与 sampler 由引用的纹理资产决定，Shader 不维护第二套 sRGB 设置。逻辑 `Color2D` 同时携带 image 和 sampler，用户不写 layout、描述符 binding 或寄存器编号。

内置 Lit 使用以下字段：

| 用途 | 字段 |
| --- | --- |
| 基础色 | `base_color`、`albedo_map`、`base_color_uv_set`、`albedo_map_st` |
| 金属 | `metallic`、`metallic_map`、`metallic_channels`、`metallic_uv_set` |
| 光滑度／导入粗糙度 | `smoothness`、`smoothness_map`、`smoothness_channels`、`smoothness_from_roughness`、`smoothness_uv_set` |
| 遮蔽 | `ambient_occlusion`、`occlusion_map`、`occlusion_strength`、`occlusion_uv_set` |
| 法线 | `normal_map`、`normal_scale`、`normal_uv_set` |
| 自发光 | `emission_color`、`emission_map`、`emission_uv_set` |

每张纹理有一个 `<name>_st` vec4，XY 是平铺，ZW 是偏移。`get/set_texture_scale` 和 `get/set_texture_offset` 使用同一个字段，不保留旧名称别名。

```python
material = inx.Material.create_lit("石材")
material.set_color("base_color", 0.6, 0.65, 0.7, 1.0)
material.set_texture("albedo_map", texture_guid)
material.set_texture_scale("albedo_map", (2.0, 2.0))
material.set_float("normal_scale", 0.8)
```

Lit 为每张贴图选择 UV0 或 UV1。UV0 法线使用模型提供的切线及手性；UV1 从位置和 UV 导数重建切线基，退化的 UV 基使用模型切线。两种路径均从 RG 编码重建法线 Z，支持导入后的双通道法线格式。

## Surface 与几何约定 {#surface-contract_1}

`SurfaceContext` 提供 `position_ws`、`normal_ws`、`tangent_ws`、`color`、`uv0`、`uv1`、`view_depth`、对象标识、layer 和 flags。`SurfaceData` 使用线性数据：

| 字段 | 含义／默认值 |
| --- | --- |
| `albedo` | 基础反射率，vec3(1) |
| `normal_ws` | 世界法线；零表示采用插值几何法线 |
| `metallic`、`smoothness` | 金属度 0，光滑度 0.5 |
| `occlusion` | 遮蔽乘数 1 |
| `emission` | 线性自发光 RGB，vec3(0) |
| `alpha` | 覆盖／不透明度 1 |
| `specular_highlights` | 直接与间接镜射乘数 1 |
| `shading_param0`、`shading_param1` | 自定义模型的两个标量载荷，0 |

同一份 Surface 求值与 alpha 决定可见颜色、阴影／深度、拾取、运动、法线和基础色通道。PBR 和 Unlit 还生成带正确模型标识的公共 GBuffer。运动通道使用上一对象变换和上一骨骼偏移。渲染状态仍属于 Material：使用其透明混合、剔除、深度、队列和 alpha 裁剪控制，不在作者文件中维护第二套状态声明。

需要局部变形时，在材质 Inspector 的顶点字段选择 `*.vert.py`，或将 `material.vert_shader_name` 设置为其 ID。顶点与片元独立选择，两个阶段的字段合入同一程序，见[顶点资产](vertex-stage.html)。

## 独立 ShadingModel {#shading-models_1}

Surface 默认采用 PBR。保存独立模型 `flat.shadingmodel.py`：

```python
import infernux.shader as shader


class Flat(shader.shadingmodel):
    shader_id = "example/shading/flat"
    hidden = True
    lighting_contract = "unlit"

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        return surface.albedo + surface.emission
```

在 Surface 类中加入 `shading_model = shader.ref("./flat.shadingmodel.py")`，也可以用 `shader.ref("example/shading/flat")` 引用已注册的 ID。alpha 仍由 Surface 持有。`ShadingContext` 包含光照所需的几何、像素位置、视深、layer 和阴影接收信息，不暴露 Mesh 专属的 UV 或切线流。模型声明 `standard_pbr`、`unlit` 或 `custom_response` 光照约定；unlit 模型不能请求光照函数。

模型内可以调用：

```python
color = shader.lighting.pbr(surface, ctx)
ambient = shader.lighting.pbr_indirect(surface, ctx)
light = shader.lighting.main_light(ctx, surface.normal_ws)
direction = shader.lighting.view_direction(ctx)
irradiance = shader.lighting.ambient(ctx, surface.normal_ws)
direct = shader.lighting.direct(surface, ctx)
```

`pbr` 包含直接光、当前环境响应和自发光。`pbr_indirect` 仅返回已经乘过材质响应的环境项，不要再次乘 albedo／BSDF 或重复叠加相同环境。它还不是未来光追的入射光输入。LightData 分开保存光色、衰减与 shadow；主光读取遵守对象 layer 和光源作用域。

`ambient` 返回尚未乘材质响应的环境辐照度。`direct` 遍历引擎的方向光、点光、聚光和矩形面光，并为每次光采样调用模型的 `shade_light`。光色、衰减和可见性保持分离；光源层、阴影视图和 Forward+ 灯光列表与 PBR 共用。需要定义以下签名：

```python
def shade_light(self, surface: shader.SurfaceData, ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
    diffuse = shader.max(shader.dot(surface.normal_ws, light.direction), 0.0)
    return surface.albedo * light.color * light.attenuation * light.shadow * diffuse
```

Slang 在编译时专门化这个接口；作者无需复制灯光循环，也不增加运行时模型判断。在 `shade` 中调用一次 `direct`，再添加所需环境响应和自发光。内置 `infernux/toon` 使用同一路径，可以独立搭配材质所选的顶点资产。

模型可调用 typed 方法和静态 ShaderLib。Surface／模型／库的依赖一起准备，并在源码成功更新后发布。内置 PBR／Unlit／Toon 参与 Deferred 的模型分发。自定义模型资产目前在 Forward／Forward+ 求颜色，包括 Deferred 管线已有的 Forward 绘制通道；自定义 GBuffer／延迟分发仍在实现。

## ShaderLib 与原始片元 {#libraries_1}

GPU 文件导入 `infernux.shader` 与明确的静态 ShaderLib 导出，不导入 NumPy、Torch、游戏组件或任意宿主模块。例如 `palette.shaderlib.py` 导出 typed 函数，相邻 Shader 用 `from .palette import quantize` 引用。内置 Surface 工具通过 `from infernux.shader.lib.surface import texture_uv, sample_normal` 引入。这些是编译依赖，不是执行 Python import。

`shader.frag` 是独立的原始片元入口：`main(self, ctx: shader.mesh_input) -> shader.vec4`，目前提供 Forward／Forward+ 颜色。引擎不会把这个最终 RGB 猜成 albedo，也不会声称它自动提供 Surface 的辅助通道。需要这些通道时使用 `shader.surface`。全屏程序使用 `shader.fullscreen`、图的 `shader.input(...)` 端口和 `shader.parameter`，共用编译及目标布局服务。

## 检查与更新 {#diagnostics_1}

`inx.Shader.check(path, source=draft)` 检查文件或未保存草稿，不发布 GPU 程序；返回有效性、逻辑 schema、SPIR-V 与 WGSL 的原文件诊断。`inx.Shader.get_schema(path)` 读取导入资产的字段。报错位置指向真正的作者、模型或库文件。宿主 import 和不支持的类型／阶段组合在准备时报告。

`inx.Shader.reload(path)` 属于制作操作。源码成功时一起发布模型／库的依赖；错误候选报告原始位置，不替换仍在使用的有效 GPU 发布。普通材质值与贴图变化只更新绑定，不重建 Shader 或渲染图。验收时同时检查实际场景、阴影、辅助通道和材质预览。
