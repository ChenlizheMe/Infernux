# LineRenderer 与 TrailRenderer 的更新方式

LineRenderer 和 TrailRenderer 共用引擎的原生线带生成、材质、按相机展开、
包围盒和网格上传路径。TrailRenderer 在这条路径上维护带时间的世界坐标轨迹。
Python 包装负责接口和 Inspector，逐帧轨迹采样由原生 LateUpdate 调度。

## 连续修改只生成一份几何

修改点、宽度、颜色和其他样式时，作者数据立即更新。引擎会合并连续修改，
在渲染提取、几何读取、包围盒读取、序列化或 bake_mesh 前生成最新网格。
不需要用户额外提交，也没有新增强制的批处理接口。

```python
line.set_positions(points)
line.start_width = 0.2
line.end_width = 0.4
line.start_color = [1, 0.5, 0, 1]
# 引擎绘制时生成这一批修改的最终几何。
```

get_position、get_positions、position_count 和宽度等属性直接读取作者数据。
vertex_count、index_count、inline_mesh_version、包围盒和烘焙需要当前几何，
会触发尚未完成的生成。若逐点编辑，请先写完，再读取这些几何结果；在每次
写点之后读取 vertex_count，仍会要求引擎逐次生成。

相同的点列表、宽度曲线和颜色渐变不会重复发布网格。静态场景缓存会在首个
有效修改时失效；空线恢复绘制也会重新进入结构收集。多相机和多次烘焙共享
已生成的数据。BakeMesh 直接读取当前原生快照，省去源顶点和索引的额外副本。

Simplify 继续使用 Ramer–Douglas–Peucker 算法，改用显式工作栈，避免长折线
导致递归调用栈溢出。算法的最坏时间复杂度没有因此改变。

## 批量坐标直接进入原生层

set_positions 和 positions 赋值接受形状为 (N, 3) 的 float32、float64 数组或
memoryview，使用本机字节序。普通列表、Vector3 列表和生成器继续可用。

```python
import numpy as np

points = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32)
line.set_positions(points)
line.set_positions(points[::-1])  # 支持有步长的视图，无需先变成连续数组
```

数组输入不再逐点创建 Python Vector3，由原生层按步长读取，并把结果交给组件
持有。只读、倒序、转置、广播和未对齐的视图也可读取。输入仍会复制到组件的
作者数据中；这不是 CPU 到 GPU 的零拷贝。修改原数组不会隐式修改线条，需要
再次调用 set_positions 才会更新。float64 最终转换为引擎坐标使用的 float32。

形状错误、整数或半精度缓冲、非本机字节序、NaN、无穷和超出 float32 范围
的坐标都会被拒绝，已有点列表保持完整。空的 (0, 3) 数组会清空线条。
TrailRenderer 的 add_positions 和 set_positions 共用这条输入通道，后者仍只
允许替换已有轨迹点，不改变历史数量。

感知颜色渐变在一次重建中只对每个关键点做一次 Oklab 转换，所有线带采样
复用转换结果。更改渐变关键点或模式后，下一次重建使用新的数据。

## 宽度和曲线

start_width 和 end_width 包含 width_multiplier。设置端点宽度会反算原始
曲线端点值，保留乘数。常规段、加权段和阶梯段继续使用共同曲线内核。
很小但 float32 可表示的宽度改动也会保留，不再被固定的近似阈值忽略。

## 轨迹与 Tween

```python
from infernux import TrailRenderer
import infernux_tween as tween

trail = emitter.add_component(TrailRenderer)
trail.time = 2.0
trail.min_vertex_distance = 0.1
tween.twn_resize(trail, 0.4, 0.1, 0.5)
tween.twn_time(trail, 1.0, 0.5)
```

轨迹自动采样使用游戏时间。emitting=False 会固定当前线头并继续老化已有点；
clear 删除历史。最小距离以下的移动仍能更新可见线头，停止发射时不会回跳。
过期边界进行插值裁剪。场景保存和克隆只保留作者配置，不保存运行轨迹。

当前引擎的 renderer-owner 约束要求每个物体使用一个渲染器。已有 MeshRenderer
的物体可使用子物体作为轨迹发射器。本轮提供了轨迹采样、存活时间及绘制所需
能力，尚未覆盖 Unity TrailRenderer 的所有其他功能，例如 autodestruct。

## 性能与验证

同一 Windows x64 Release 构建环境，逐点修改整条线，再改两端宽度，最后读取
vertex_count。每组四次，表中为中位数，包含 Python 调用及最终网格生成：

| 点数 | 修改前 CPU 耗时 | 修改后 CPU 耗时 | 几何发布次数 |
| --- | --- | --- | --- |
| 256 | 6.42 ms | 0.20 ms | 258 → 1 |
| 1,024 | 103.14 ms | 0.78 ms | 1,026 → 1 |
| 4,096 | 1,971.71 ms | 3.04 ms | 4,098 → 1 |

这是特定更新方式的 CPU 测量，不代表游戏帧率或 GPU 加速比例。
复现工具为 tests/tools/benchmark_line_edits.py；本地原始记录保存在
tests/fixtures/line_renderer/edit_batch_windows.json。

进一步优化数组输入和感知颜色渐变后，同一 Windows x64 Release 环境的中位数
如下。每组预热三次、测量十五次，每次都修改数据并读取 vertex_count，包含
最终网格生成；“修改前”已经包含上表中的连续修改合并优化：

| 点数 | 数组更新修改前 | 数组更新修改后 | 感知渐变重建修改前 | 感知渐变重建修改后 |
| --- | --- | --- | --- | --- |
| 1,024 | 0.690 ms | 0.096 ms | 0.139 ms | 0.091 ms |
| 4,096 | 2.977 ms | 0.379 ms | 0.566 ms | 0.357 ms |
| 16,384 | 12.798 ms | 2.101 ms | 2.768 ms | 2.071 ms |

数组更新使用公共 Python LineRenderer.set_positions；感知渐变重建只修改一个
原生点，使用两个颜色关键点，用来分开观察网格生成中的计算开销。复现工具为
tests/tools/benchmark_line_batch.py，完整样本记录在
tests/fixtures/line_renderer/array_update_windows.json。以上仍是 CPU 测量，
不代表帧率或 GPU 性能提升。

测试覆盖修改合并、无变化更新、读取最新包围盒、渲染前烘焙、多相机烘焙、
长折线简化及浮点小宽度。实际 GPU 读回验证静态线移动、移出视野、清空、
退化成重合点、恢复和宽度变化。轨迹还有真实 Tween 插件到 GPU 的联动验证。
原有透视、正交、自定义视图、近乎平行视角、局部坐标、镜像和 Transform Z
烘焙像素对照也通过了回归。

新增代码通过 Android ARM64 编译检查、WebAssembly 编译和 Linux 编译检查。
这些结果不替代 Android/Web 完整 Player 的端到端验收。
数组通道及感知渐变优化另外通过了 54 项相关测试、8 项实时绘制/烘焙与裁剪
测试，以及独立的八阶段 GPU 读回验证。修改后的 LineRenderer 核心重新通过
Linux 语法检查和 WebAssembly 编译；数组 Python 绑定目前在 Windows 验证。
