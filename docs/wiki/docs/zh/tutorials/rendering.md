# 渲染与后处理 — 让画面更好看

<div class="class-info">
教程 &nbsp;|&nbsp; <a href="../../en/tutorials/rendering.html">English</a>
</div>

## 概述

InfEngine 配备了一套 **可编程渲染管线**（Scriptable Render Pipeline）系统，灵感来自 Unity URP/HDRP。你可以选择内置管线、调整参数、叠加后处理效果，或者像疯狂科学家一样从零编写自定义管线。全部用 Python。不用重新编译 shader（至少管线部分不用）。

架构分三层：

| 层级 | 职责 | 什么时候需要动它 |
|---|---|---|
| **RenderPipeline** | 定义*拓扑结构* — 哪些 Pass 按什么顺序执行 | 想要完全自定义管线时 |
| **RenderStack** | 管理一个管线 + 可注入的 Pass。挂在 GameObject 上 | 想要在运行时增删/排序效果时 |
| **FullScreenEffect** | 多 Pass 全屏后处理效果（Bloom、Vignette 等） | 想要好看的画面时 |

## 快速上手：60 秒搞定后处理

最快让后处理跑起来的方法：

```python
from InfEngine import *
from InfEngine.renderstack import *

class SetupPostProcessing(InfComponent):
    def start(self):
        # RenderStack 是组件 — 挂到任何 GameObject 上
        stack = self.game_object.add_component(RenderStack)
        # 默认使用 DefaultForwardPipeline
        
        # 加几个效果
        bloom = BloomEffect()
        bloom.threshold = 1.0
        bloom.intensity = 0.8
        bloom.scatter = 0.7
        stack.add_pass(bloom)
        
        vignette = VignetteEffect()
        vignette.intensity = 0.4
        stack.add_pass(vignette)
        
        # 色调映射放最后（HDR → LDR）
        tonemapping = ToneMappingEffect()
        tonemapping.exposure = 1.2
        stack.add_pass(tonemapping)
```

搞定。你的场景现在有了泛光、暗角和色调映射。就是这么简单。

## 内置管线

### Default Forward（默认前向渲染）

主力管线。先渲染不透明物体（从前到后，利用 early-z），再画天空盒，最后画透明物体（从后到前混合）。支持阴影贴图和 MSAA。

```
ShadowCasterPass → OpaquePass → [after_opaque]
→ SkyboxPass → [after_sky]
→ TransparentPass → [after_transparent]
→ [before_post_process] → ScreenUI → [after_post_process]
```

Inspector 中暴露的参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `shadow_resolution` | 4096 | 阴影贴图尺寸（256–8192） |
| `msaa_samples` | X4 | 抗锯齿：OFF、X2、X4、X8 |
| `enable_screen_ui` | True | 是否渲染屏幕空间 UI |

### Default Deferred（默认延迟渲染）

适合光源多的场景。使用 GBuffer 写入 4 个 MRT（albedo、法线、材质属性、自发光）+ 深度，然后在全屏 Pass 中计算光照。

```
ShadowCasterPass → GBufferPass (MRT) → [after_gbuffer]
→ DeferredLightingPass → [after_opaque]
→ SkyboxPass → [after_sky]
→ TransparentPass（前向渲染） → [after_transparent]
→ [before_post_process] → ScreenUI → [after_post_process]
```

> **注意：** 延迟管线需要 `deferred_lighting` shader。如果没有，它会输出原始 albedo。

### 切换管线

```python
stack = game_object.get_py_component(RenderStack)
stack.set_pipeline("Default Deferred")
# 或者
stack.set_pipeline("Default Forward")
```

## 后处理效果

所有内置效果都继承自 `FullScreenEffect`，通过 `serialized_field` 声明参数，可在 Inspector 中调节。

### Bloom（泛光）

经典的"亮的地方会发光"效果。使用逐级降采样/升采样链。

```python
bloom = BloomEffect()
bloom.threshold = 1.0      # 亮度阈值，超过才会泛光
bloom.intensity = 0.8      # 最终泛光强度
bloom.scatter = 0.7        # 扩散范围（0–1）
bloom.clamp = 65472.0      # 亮度上限（防止萤火虫噪点）
bloom.tint_r = 1.0         # 着色（默认白色）
bloom.tint_g = 1.0
bloom.tint_b = 1.0
bloom.max_iterations = 5   # 降采样次数（1–8）
stack.add_pass(bloom)
```

**注入点：** `before_post_process`（顺序 100）

### Color Adjustments（颜色调整）

曝光、对比度、饱和度、色相偏移。"一键电影感"旋钮。

```python
ca = ColorAdjustmentsEffect()
ca.post_exposure = 0.5   # EV 曝光偏移（-5 到 5）
ca.contrast = 10.0       # -100 到 100
ca.saturation = -20.0    # -100 到 100（负数 = 去饱和）
ca.hue_shift = 0.0       # -180 到 180 度
stack.add_pass(ca)
```

**注入点：** `before_post_process`（顺序 200）

### White Balance（白平衡）

色温和色调。让你的场景感觉温暖舒适，或者冷冽末世。

```python
wb = WhiteBalanceEffect()
wb.temperature = 15.0   # 正值 = 暖色，负值 = 冷色（-100 到 100）
wb.tint = 0.0           # 绿/品红偏移（-100 到 100）
stack.add_pass(wb)
```

**注入点：** `before_post_process`（顺序 150）

### Vignette（暗角）

让画面边缘变暗，瞬间"电影感"。

```python
vig = VignetteEffect()
vig.intensity = 0.35    # 0 = 关闭，1 = 全黑边缘
vig.smoothness = 0.3    # 衰减柔和度
vig.roundness = 1.0     # 1 = 圆形，越小越方
vig.rounded = False     # 强制正圆（不管宽高比）
stack.add_pass(vig)
```

**注入点：** `before_post_process`（顺序 500）

### Chromatic Aberration（色差/色散）

RGB 通道分离。轻微 = "电影镜头感"；严重 = "主角刚挨了一拳"。

```python
chroma = ChromaticAberrationEffect()
chroma.intensity = 0.1   # 0 = 关闭，1 = 强烈分离
stack.add_pass(chroma)
```

**注入点：** `before_post_process`（顺序 600）

### Film Grain（胶片颗粒）

电影噪点。让你的游戏看起来像 90 年代的电影。或者 2020 年代的——因为复古又流行了。

```python
grain = FilmGrainEffect()
grain.intensity = 0.2    # 0 = 关闭，1 = 重度噪点
grain.response = 0.8     # 0 = 均匀，1 = 仅影响高光
stack.add_pass(grain)
```

**注入点：** `after_post_process`（顺序 800）

### Sharpen（锐化）

对比度自适应锐化（AMD FidelityFX CAS）。增强细节，不会产生光晕。

```python
sharp = SharpenEffect()
sharp.intensity = 0.5    # 0 = 关闭，1 = 最大
stack.add_pass(sharp)
```

**注入点：** `after_post_process`（顺序 850）

### Tone Mapping（色调映射）

将 HDR 转换为可显示的 LDR。**应该是 HDR 链中最后一个效果。**

```python
tm = ToneMappingEffect()
tm.mode = ToneMappingMode.ACES   # ACES、Reinhard 或 None_
tm.exposure = 1.0                 # 映射前曝光倍数
tm.gamma = 2.2                    # 伽马校正（2.2 = sRGB）
stack.add_pass(tm)
```

**注入点：** `after_post_process`（顺序 900）

### 推荐效果叠加顺序

效果按各自注入点内的 `default_order` 排序。典型顺序：

```
before_post_process（HDR 空间）:
  100  Bloom           — 提取亮区
  150  White Balance   — 调色温
  200  Color Adj.      — 曝光/对比度/饱和度
  500  Vignette        — 暗角（仍在 HDR）
  600  Chrom. Aberr.   — 色散

after_post_process（LDR 空间）:
  800  Film Grain      — 噪点
  850  Sharpen         — CAS 锐化
  900  Tone Mapping    — HDR → LDR 转换
```

## RenderGraph（渲染图）

底层来说，每个管线都构建一个 `RenderGraph` — 一个由纹理资源和渲染 Pass 组成的有向无环图（DAG）。基础后处理不需要碰它，但自定义管线时它非常强大。

### 纹理

```python
from InfEngine.rendergraph import RenderGraph, Format

graph = RenderGraph("MyGraph")

# 跟相机同尺寸的颜色目标
color = graph.create_texture("color", camera_target=True)

# 深度缓冲
depth = graph.create_texture("depth", format=Format.D32_SFLOAT)

# 半分辨率纹理（用于降采样）
half_color = graph.create_texture("half_color",
    format=Format.RGBA16_SFLOAT,
    size_divisor=2)

# 固定尺寸纹理（如阴影贴图）
shadow = graph.create_texture("shadow_map",
    format=Format.D32_SFLOAT,
    size=(4096, 4096))
```

### Pass

```python
# 上下文管理器风格（推荐）
with graph.add_pass("OpaquePass") as p:
    p.write_color("color")
    p.write_depth("depth")
    p.set_clear(color=(0.1, 0.1, 0.1, 1.0), depth=1.0)
    p.set_texture("shadowMap", "shadow_map")
    p.draw_renderers(queue_range=(0, 2500), sort_mode="front_to_back")

# 全屏四边形 Pass（用于后处理）
with graph.add_pass("MyEffect") as p:
    p.set_texture("_SourceTex", color)
    p.write_color(output)
    p.fullscreen_quad("my_shader")
```

### 注入点

注入点是命名的"插槽"，外部 Pass 可以在这里插入：

```python
graph.injection_point("after_opaque", resources={"color", "depth"})
```

`resources` 集合告诉系统该点上有哪些纹理可用。

## 编写自定义管线

想要完全掌控？继承 `RenderPipeline`：

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
        
        # 添加注入点让效果可以接入
        graph.injection_point("before_post_process", resources={"color"})
        graph.injection_point("after_post_process", resources={"color"})
        
        graph.set_output("color")
```

然后使用：

```python
stack = game_object.get_py_component(RenderStack)
stack.set_pipeline("Minimal Forward")
```

## 编写自定义 FullScreenEffect

想加自己的后处理？继承 `FullScreenEffect`：

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

自定义效果的关键规则：
1. 从 `bus.get("color")` 读取输入（或你需要的其他资源）
2. 创建一个**新的**输出纹理（不要写回输入纹理——会产生读写冲突！）
3. 把结果写回 `bus.set("color", output)`
4. `get_or_create_texture` 辅助方法避免重复创建纹理

## 运行时效果控制

```python
stack = game_object.get_py_component(RenderStack)

# 开关效果
stack.set_pass_enabled("Bloom", False)
stack.set_pass_enabled("Bloom", True)

# 完全移除效果
stack.remove_pass("Film Grain")

# 改变效果顺序
stack.reorder_pass("Vignette", 50)  # 数字越小越早执行

# 将一个 Pass 移到另一个之前
stack.move_pass_before("Chromatic Aberration", "Vignette")

# 修改后强制重建渲染图
stack.invalidate_graph()
```

## ResourceBus（资源总线）

`ResourceBus` 是渲染 Pass 之间通信的方式。本质上是一个 字符串 → TextureHandle 的字典：

```python
def setup_passes(self, graph, bus):
    # 读取上一个效果产出的结果
    color = bus.get("color")       # TextureHandle 或 None
    depth = bus.get("depth")       # 如果没有深度 Pass 写入，可能为 None
    
    # ... 做你的事 ...
    
    # 写回，让下一个效果拿到你的输出
    bus.set("color", my_output)
```

可用资源取决于注入点。通常：
- `before_post_process`：`color`、`depth`
- `after_post_process`：`color`（深度可能不可用）
- 自定义注入点：管线声明了什么就有什么

## 发现可用管线和 Pass

```python
from InfEngine.renderstack import discover_pipelines, discover_passes

# 所有已注册的管线类
pipelines = discover_pipelines()
for name, cls in pipelines.items():
    print(f"  {name}: {cls}")

# 所有已注册的渲染 Pass 类
passes = discover_passes()
for name, cls in passes.items():
    print(f"  {name}: {cls}")
```

## 小贴士

- **效果顺序很重要。** 在 HDR 空间做 Bloom 和在色调映射后做 Bloom 效果完全不同。内置的 `default_order` 给你合理的默认值。
- **修改 Stack 后记得调用 `invalidate_graph()`**。渲染图为了性能会被缓存。
- **色调映射应该放在最后**（至少在所有 HDR 效果之后）。否则 Bloom 操作的是已经压缩过的值，看起来会很灰。
- **读写冲突**：永远不要对正在读取的纹理做 `write_color`。必须创建新的输出纹理。
- **性能**：每个效果至少增加一个全屏 Pass。在低端硬件上，少即是多。

## 另请参阅

- [RenderGraph API](../api/RenderGraph.md)
- [RenderStack API](../api/RenderStack.md)
- [FullScreenEffect API](../api/FullScreenEffect.md)
- [BloomEffect API](../api/BloomEffect.md)
- [ToneMappingEffect API](../api/ToneMappingEffect.md)
- [RenderPipeline API](../api/RenderPipeline.md)
