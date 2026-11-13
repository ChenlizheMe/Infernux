<!-- language:en -->

<span class="mini-tag">Custom Rendering · Chapter 4</span>

# Shading models and static shader libraries

A `.frag.py` surface describes the material at a point. A `.shadingmodel.py` evaluates its response to light. A `.shaderlib.py` holds typed functions shared by GPU assets. These are static GPU source files: the engine reads their syntax and Slang compiles their programs. They are not imported as ordinary Python components.

This chapter describes the Slang rendering development branch. The surface, model, and library below work together in Forward and Forward+. The current Deferred path has a separate model-registration boundary described below.

## Build a complete three-file model {#complete-example}

Create this folder:

```text
Assets/Shaders/bands/
  band_math.shaderlib.py
  bands.shadingmodel.py
  painted.frag.py
```

Put this function in `band_math.shaderlib.py`:

```python
import infernux.shader as shader


def band(value: float, threshold: float, softness: float) -> float:
    width = shader.max(softness, 0.0001)
    return shader.smoothstep(threshold - width, threshold + width, value)
```

Put the light response in `bands.shadingmodel.py`:

```python
import infernux.shader as shader
from .band_math import band


class Bands(shader.shadingmodel):
    shader_id = "my_game/shading/bands"
    hidden = True
    lighting_contract = "custom_response"

    def shade_light(self, surface: shader.SurfaceData,
                    ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
        normal = shader.normalize(surface.normal_ws)
        value = shader.max(shader.dot(normal, light.direction), 0.0)
        response = band(value, surface.shading_param0, surface.shading_param1)
        radiance = light.color * light.attenuation * light.shadow
        return surface.albedo * response * radiance

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        normal = shader.normalize(surface.normal_ws)
        ambient = shader.lighting.ambient(ctx, normal) * surface.albedo * surface.occlusion
        return shader.lighting.direct(surface, ctx) + ambient + surface.emission
```

`shade_light` describes one direct-light response. `lighting.direct` traverses the engine's eligible light set and calls it; authors do not access light-list descriptor slots. Light direction, color, distance attenuation, and shadow visibility are separate values. `shade` combines direct, ambient, and emission terms.

Put the material-facing surface in `painted.frag.py`:

```python
import infernux.shader as shader


class Painted(shader.surface):
    shader_id = "my_game/materials/bands"
    shading_model = shader.ref("bands.shadingmodel.py")
    base_color = shader.material.color(default=(0.8, 0.3, 0.1, 1.0))
    threshold = shader.material.float(default=0.5, range=(0.0, 1.0))
    softness = shader.material.float(default=0.04, range=(0.0, 0.25))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(
            albedo=self.base_color.rgb,
            normal_ws=ctx.normal_ws,
            alpha=self.base_color.a,
            shading_param0=self.threshold,
            shading_param1=self.softness,
        )
```

Import the files, select `my_game/materials/bands` as a Material's fragment stage, and retain `infernux/standard` as its vertex stage. Adjust Threshold and Softness in the Inspector. The fields are sliders because their declarations include `range`; they remain ordinary scalar material properties. The fragment does not select or import a vertex asset.

## Keep the data contracts separate {#design-boundary}

| Type | Owns |
| --- | --- |
| `SurfaceContext` | Mesh position, normals, tangents, UVs, vertex color, and object data supplied by the engine. |
| `SurfaceData` | Albedo, shading normal, metallic, smoothness, occlusion, emission, alpha, specular control, and two model-defined scalar channels. |
| `ShadingContext` | Lighting geometry: position, geometric normal, view depth, pixel position, layer mask, and shadow policy. |
| `LightData` | One incident light's direction, color, attenuation, and visibility. |

A shading model consumes `SurfaceData`; it should not sample a particular material's private texture or depend on a particular mesh's UVs. The surface performs those operations before lighting. The two `shading_param` channels carry per-material model controls in the example.

## Select a lighting contract {#requirements}

| Contract | Meaning |
| --- | --- |
| `standard_pbr` | Use the engine's PBR response and shared light/environment inputs. The built-in model delegates to `shader.lighting.pbr(surface, ctx)`. |
| `unlit` | Return surface color and emission without requesting scene-light resources. The built-in model returns `surface.albedo + surface.emission`. |
| `custom_response` | Define a typed lighting response, including `shade_light` when calling `lighting.direct`. |

The contract is a compile-time statement of the response's requirements. It is not a request to guess surface data from the final RGB output.

The current Deferred evaluator registers the built-in PBR, Unlit, and Toon responses. An arbitrary project model is not registered merely because it declares `custom_response` or has a Shader ID. Route such a model through Forward or Forward+ until it has a corresponding Deferred registration and a compatible GBuffer representation. A pipeline can mix these routes. Renaming a custom model to a built-in ID does not add that support.

## Import a library without running host code {#libraries}

Relative imports resolve `.shaderlib.py` files from the GPU asset's location. Engine libraries use paths such as `from infernux.shader.lib.color import luminance`. Functions use typed positional arguments and a typed return value. GPU resources can be passed through typed library parameters; the compiler owns target bindings.

Only the GPU vocabulary and static ShaderLib imports belong here. NumPy, component lifecycle, Python reflection, file IO, and arbitrary host-library execution belong in ordinary component or tool scripts. `shader.frame_time()` reads the engine-owned render clock; it does not ask Python to upload a separate clock.

## Diagnose and publish changes {#discovery-diagnostics}

In a host tool, `Shader.check(path)` checks the source and its dependencies without publishing a revision or creating a graphics pipeline. Its diagnostics retain the author file and location. `Shader.get_schema(path)` reports parameters and presentation metadata. Importing or reloading publishes a successful asset revision; a failed edit does not replace an already published program. Console tracks active import/compile failures and resolves them after the corresponding successful operation.

Check a changed library through a consuming fragment as well as on its own: the final stage combination determines which resources and entry points are needed. Keep the vertex and fragment selections independent, and check the actual pipeline route in which the Material will be drawn.

<!-- language:zh -->

<span class="mini-tag">自定义渲染 · 第 4 章</span>

# ShadingModel 与静态 ShaderLib

`.frag.py` 中的 Surface 描述一个点的材质，`.shadingmodel.py` 计算表面对光的响应，`.shaderlib.py` 保存 GPU 资源之间复用的类型化函数。这些都是静态 GPU 源码：引擎读取语法，再由 Slang 编译；它们不作为普通 Python 组件导入。

本章对应 Slang 渲染开发分支。下面的三文件示例可组合用于 Forward 和 Forward+。目前 Deferred 另有模型登记边界，后文会说明。

## 完整的三文件示例 {#complete-example_zh}

创建以下目录：

```text
Assets/Shaders/bands/
  band_math.shaderlib.py
  bands.shadingmodel.py
  painted.frag.py
```

在 `band_math.shaderlib.py` 中写一个可复用函数：

```python
import infernux.shader as shader


def band(value: float, threshold: float, softness: float) -> float:
    width = shader.max(softness, 0.0001)
    return shader.smoothstep(threshold - width, threshold + width, value)
```

在 `bands.shadingmodel.py` 中定义光照响应：

```python
import infernux.shader as shader
from .band_math import band


class Bands(shader.shadingmodel):
    shader_id = "my_game/shading/bands"
    hidden = True
    lighting_contract = "custom_response"

    def shade_light(self, surface: shader.SurfaceData,
                    ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
        normal = shader.normalize(surface.normal_ws)
        value = shader.max(shader.dot(normal, light.direction), 0.0)
        response = band(value, surface.shading_param0, surface.shading_param1)
        radiance = light.color * light.attenuation * light.shadow
        return surface.albedo * response * radiance

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        normal = shader.normalize(surface.normal_ws)
        ambient = shader.lighting.ambient(ctx, normal) * surface.albedo * surface.occlusion
        return shader.lighting.direct(surface, ctx) + ambient + surface.emission
```

`shade_light` 定义单个直接光的响应，`lighting.direct` 遍历引擎筛选出的光源并调用它。作者不用声明光源列表的描述符位置。光的方向、颜色、距离衰减和阴影可见性分别提供；`shade` 再组合直接光、环境项和发光。

在 `painted.frag.py` 中提供面向材质的 Surface：

```python
import infernux.shader as shader


class Painted(shader.surface):
    shader_id = "my_game/materials/bands"
    shading_model = shader.ref("bands.shadingmodel.py")
    base_color = shader.material.color(default=(0.8, 0.3, 0.1, 1.0))
    threshold = shader.material.float(default=0.5, range=(0.0, 1.0))
    softness = shader.material.float(default=0.04, range=(0.0, 0.25))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(
            albedo=self.base_color.rgb,
            normal_ws=ctx.normal_ws,
            alpha=self.base_color.a,
            shading_param0=self.threshold,
            shading_param1=self.softness,
        )
```

导入文件，在材质中选择 `my_game/materials/bands` 作为片段资源，顶点资源保留 `infernux/standard`，再调节 Inspector 中的 Threshold 和 Softness。字段声明了 `range`，因此显示为滑条；它们仍是普通的材质标量属性。片段文件不选择或导入顶点资源。

## 分清几种数据 {#design-boundary_zh}

| 类型 | 保存什么 |
| --- | --- |
| `SurfaceContext` | 引擎提供的网格位置、法线、切线、UV、顶点颜色和物体信息。 |
| `SurfaceData` | 反照率、着色法线、金属度、光滑度、遮蔽、发光、透明度、高光开关和两个模型自定义标量通道。 |
| `ShadingContext` | 光照几何信息，包括位置、几何法线、视图深度、像素位置、层掩码和阴影策略。 |
| `LightData` | 单个入射光的方向、颜色、衰减和可见性。 |

ShadingModel 消费 `SurfaceData`，不应采样某个材质的私有贴图，也不依赖某个网格的 UV。这些操作由 Surface 在光照之前完成。示例用两个 `shading_param` 通道把材质控制量传给模型。

## 选择光照合同 {#requirements_zh}

| 合同 | 含义 |
| --- | --- |
| `standard_pbr` | 使用引擎 PBR 响应和共享的光源、环境输入。内置模型调用 `shader.lighting.pbr(surface, ctx)`。 |
| `unlit` | 不请求场景光照资源，直接输出表面颜色与发光。内置模型返回 `surface.albedo + surface.emission`。 |
| `custom_response` | 定义类型化的光照响应；调用 `lighting.direct` 时提供 `shade_light`。 |

合同在编译期说明响应需要哪些能力，不负责从最终 RGB 反推表面信息。

目前 Deferred 求值器登记了内置 PBR、Unlit 和 Toon 响应。项目模型声明 `custom_response` 或 Shader ID，不会自动完成 Deferred 登记。在模型有对应的 Deferred 实现与兼容 GBuffer 表达之前，应走 Forward 或 Forward+；一条管线可以混合这些路由。把项目模型改名为内置 ID 不能补齐这项能力。

## 导入库，不执行宿主代码 {#libraries_zh}

相对导入从 GPU 资产所在目录解析 `.shaderlib.py`。引擎函数库可使用 `from infernux.shader.lib.color import luminance` 这样的路径。函数的参数与返回值都需要类型；类型化资源也可以在库函数之间传递，目标绑定由编译器安排。

这里使用 GPU 词表和静态 ShaderLib 导入。NumPy、组件生命周期、Python 反射、文件读写和任意宿主库执行应放在普通组件或工具脚本中。`shader.frame_time()` 读取引擎统一维护的渲染时钟，不要求 Python 额外上传一份时间。

## 检查与发布 {#discovery-diagnostics_zh}

宿主工具中的 `Shader.check(path)` 检查源码和依赖，不发布修订，也不创建图形管线；诊断保留作者文件与位置。`Shader.get_schema(path)` 返回参数与显示声明。成功导入或重载之后，新资产修订才会发布；失败编辑不会覆盖已经发布的程序。Console 跟踪活动的导入、编译问题，并在对应操作成功之后解决。

修改库之后，还要通过使用它的片段资源进行检查：最终阶段组合决定需要哪些资源与入口。保持顶点和片段选择独立，并验证材质实际使用的渲染路由。
