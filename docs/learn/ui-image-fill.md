# 图像填充与 Tween

UIImage 现在支持纹理图像的水平、垂直和径向填充。先在 Inspector 中指定纹理，
再将图像类型设为“填充”。普通图像继续使用原来的绘制路径。

```python
from infernux.ui import UIImage

image = game_object.get_component(UIImage)
image.image_type = image.Type.FILLED
image.fill_method = image.FillMethod.RADIAL360
image.fill_origin = 0
image.fill_clockwise = True
image.fill_amount = 0.25
```

先设置方式，再设置起点。修改 `fill_method` 会将 `fill_origin` 重置为零；这与
Unity 的属性行为相同。Inspector 使用与其他组件相同的字段控件，并按当前方式
显示起点名称。一次方式修改及其起点变化可以完整撤销、重做。

| 填充方式 | 起点编号 |
| --- | --- |
| `HORIZONTAL` | 0 左，1 右 |
| `VERTICAL` | 0 下，1 上 |
| `RADIAL90` | 0 左下，1 左上，2 右上，3 右下 |
| `RADIAL180` | 0 下，1 左，2 上，3 右 |
| `RADIAL360` | 0 下，1 右，2 上，3 左 |

`fill_amount` 钳制在 0 到 1；NaN 会被拒绝。径向填充还使用 `fill_clockwise`。
UV 按原图裁切，保留下来的部分不会重新拉伸整张纹理。旋转和镜像以完整图像
的中心计算，因此填充进度不会改变旋转中心。当前 `ImageType` 提供 Simple 和
Filled；这不表示 Sliced、Tiled 或完整 Unity Sprite 系统已经实现。

未指定纹理时，UIImage 继续绘制原来的完整背景矩形，填充量不裁切背景。
这与 Unity Image 没有 Sprite 时回到普通矩形的行为相同。默认命中区域也仍然
是完整矩形，填充为零不意味着关闭交互；要关闭交互，设置 `raycast_target=False`。

安装对应版本的 Tween 插件后，可以直接补间填充量：

```python
import infernux_tween as tween

tween.twn_fill_amount(image, 0.8, 0.3).set_ease(tween.Ease.IN_OUT_SINE)
```

快捷函数先钳制终值，再创建补间；属性写入仍会钳制采样结果，因此自定义缓动
超出 0 到 1 时也不会生成越界填充。From、相对模式和可视化 Tween Animation 的
`FILL_AMOUNT` 使用同一接口。编辑器预览结束会还原原填充量。

图像几何由同一个 C++ 裁剪内核生成，编辑器与运行时共用，不增加独立 DLL、
平台专属着色器或 CPU 纹理读取。内核使用有界栈内存，继续通过现有 UI 绘制列表
管理顶点、纹理、材质和 World UI。诊断用几何接口在计算期间释放 GIL；实际
ImGui 绘制操作继续遵守渲染线程的状态约束。

当前验证范围：

- 真实 Unity 2022.3 Image 导出的 672 组网格，核对填充面积、区域和 UV，包含非正方形矩形、全部起点/方向及边界进度。
- 673 项内核与默认值检查、101 项相关引擎 UI 回归；这两组测试不能与其他历史测试数字直接相加。
- 编辑器原生绘制验证 UV、旋转中心、镜像、纹理状态还原及 16 位顶点索引跨界。
- 13 组真实 Canvas GPU 读回，覆盖五种方式、旋转、镜像、空/满进度和无纹理行为；另一次端到端运行通过 Tween 的中间采样驱动相同画面。
- 快捷函数和作者组件覆盖 From、相对模式和超范围终值；插件另有无效输入的创建原子性及预览还原检查。
- Linux 原生执行、Android ARM64 真机执行、WebAssembly/Node 执行相同内核；Web 还编译了共享绘制代码。

这些平台探针不替代完整 Linux、Android、Web Player 的绘制与插件安装验收。
新能力需要包含对应原生接口的引擎 SDK；旧的 changeable runtime 没有同步修改。
