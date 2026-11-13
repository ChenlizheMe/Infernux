# Shader

<div class="class-info">
类位于 <b>infernux.core</b>
</div>

## 描述

查询、检查和重新导入 GPU 着色器资源。

<!-- USER CONTENT START --> description

本页描述当前 Slang 渲染开发分支的接口。`infernux.core.Shader` 是宿主工具接口；`infernux.shader` 是 GPU 作者文件使用的声明与数学词表。

着色器写在独立的 `.frag.py`、`.vert.py`、`.shadingmodel.py` 和 `.shaderlib.py` 资源中。引擎静态分析这些文件并通过 Slang 生成目标程序。作者文件不作为普通组件模块导入，其中的函数在 GPU 上执行。

资源通过 `shader_id` 声明逻辑名称，例如 `infernux/lit`。材质分别选择顶点和片段资源，选择器按名称中的 `/` 形成多级菜单。片段源不引用或固定一个顶点资源。资产 GUID 持有引用身份，逻辑名称负责查询和显示。

<!-- USER CONTENT END -->

## 静态方法

| 方法 | 描述 |
|------|------|
| `Shader.check(asset: str, *, source: str \| None = None) → dict` | 检查资源或未保存的草稿，返回逻辑结构与两个 GPU 目标的诊断。 |
| `Shader.get_schema(asset: str) → dict` | 查询逻辑字段、图输入和资源域，不创建设备管线。 |
| `Shader.property_to_id(name: str) → int` | 获得当前进程内复用的材质属性 ID。 |
| `Shader.is_loaded(name: str, shader_type: str = "vertex") → bool` | 查询阶段或已链接材质程序是否已发布到渲染器。 |
| `Shader.reload(shader_id: str, shader_type: str \| None = None) → bool` | 经资产系统重新导入 Editor 中的着色器源码。 |

<!-- USER CONTENT START --> static_methods

### 检查与发布

`check` 接受资源路径、GUID 或已登记的唯一 Shader ID。传入 `source` 可以检查尚未保存的草稿。结果包含 `valid`、`schema` 和 `diagnostics`，诊断带源文件与位置。这个操作不写入资源、不发布修改，也不创建图形管线。

`get_schema` 返回当前修订的逻辑结构，包括 `domain`、`info`、`parameters` 与 `inputs`。参数可带 `range`、`label`、`hdr` 等信息，Inspector 使用这些声明。浮点字段的 `range=(0.0, 1.0)` 会生成范围滑条。

`reload` 用于 Editor 或无界面的创作宿主。它按资源路径或逻辑名称执行重新导入，失败抛出异常。按顶点、片段名称查询时，可指定 `shader_type="vertex"` 或 `"fragment"`；同一阶段重名时使用路径。多文件重新导入按顺序执行。

`is_loaded` 只查询已发布状态，不触发加载。宿主尚未启动、已关闭或没有渲染器时返回 `False`。失败编辑之后，上一次可用的程序仍可能处于已发布状态，所以它不表示最近一次编辑编译成功。草稿使用 `check`；实际导入错误由 Console 跟踪，修复并成功发布后解决。

Player 的编译器保持可用，以准备运行所需的新程序组合或计算签名；冻结的创作资源不能通过 `reload` 原地修改。`property_to_id` 返回进程内 ID，适合重复的材质属性更新；持久文件保存属性名称。

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

创建 `Assets/Shaders/warm_surface.frag.py`：

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

这是静态 GPU 源码。只使用引擎 GPU 词表和支持的 ShaderLib 导入；不在这里导入 NumPy，也不放组件生命周期或宿主 IO。

导入后，在普通 Python 工具或组件代码中创建材质：

```python
from infernux.core import Material, Shader

path = "Assets/Shaders/warm_surface.frag.py"
report = Shader.check(path)
print(report["valid"], report["diagnostics"])

material = Material.from_shader(path, "暖色材质")
smoothness_id = Shader.property_to_id("smoothness")
material.set_float(smoothness_id, 0.7)
```

材质从 `infernux/standard` 顶点阶段开始，在 Inspector 中可以独立选择适配的顶点资源。引擎准备同一表面所需的前向、深度、阴影、运动等入口，作者不逐个复制表面逻辑。

作者类的 `hidden = True` 会从普通选择菜单隐藏资源，其 ID 和现有引用仍然有效。

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

[Material](Material.md) · [顶点阶段](../../../../learn/vertex-stage.html) · [片元与材质](../../../../learn/fragment-materials.html)

<!-- USER CONTENT END -->
