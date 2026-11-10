# UI 布局与运行时位置

UI 的矩形、场景 Transform、绘制和点击检测使用同一份原生矩形状态。容器布局产生的实际尺寸与位置是运行时派生结果；用户设置的尺寸、锚点和位置是创作输入。保存场景和复制对象时使用创作输入，避免把窗口大小或一次布局的结果写回资产。

## 自动发布

运行时在读取 UI 几何依赖和提交绘制命令前，更新使用原生矩形或自动布局的 Canvas 与世界 UI。流程先分配子元素宽度，再根据实际宽度测量换行文字，最后发布文字尺寸和受其影响的排列。没有布局输入、成员、可见性或视口变化时，复用已发布的布局。

`UIFrame` 控制 Flow 子元素的位置与尺寸。`UIText` 的自动尺寸只控制所需的轴：`AutoWidth` 控制宽度，`AutoHeight` 控制高度。参与 Flow 的文字由父容器统一分配矩形。容器关闭布局、子元素离开布局或驱动释放后，恢复对应的创作输入；无关的轴和 Z 位置不受尺寸驱动影响。

手动工具可以主动发布布局：

```python
canvas.rebuild_layout(logical_width, logical_height)
frame.rebuild_layout()  # 屏幕 UI 更新所属 Canvas；世界 UI 更新本子树。
```

这里的宽高是 Canvas 的逻辑尺寸。运行时通过已有的 CanvasScaler 计算逻辑尺寸与文字缩放，无须再手工转换像素。

Canvas 挂接到对象时建立原生参考矩形。拉伸控件换到一个新的屏幕 Canvas 后，可以立即读取新父级下的尺寸，不需要等首次绘制。正在播放的动画继续操作原来的矩形所有者。

新父级可以包含尚未绑定矩形的普通空物体或 UI 容器。原生修改父级时只准备这条必要的祖先链，读取容器已有尺寸，并让空物体传递父矩形；不把父尺寸暂时当作零。`GameObject.set_parent`、`Transform.set_parent` 和 `Transform.parent` 使用同一条路径，保持世界姿态时仍保留矩形大小。稳定播放不重复初始化父级，未参与动画的兄弟控件保持原来的数据和绑定状态。

UI 矩形的逻辑坐标使用像素；世界 UI 通过每 100 像素对应 1 个场景单位换算。矩形随父级在屏幕与世界空间之间切换时，由原生层更新换算，包含 Z 坐标和已绑定的子树。`set_parent(parent, False)` 保留逻辑锚点坐标，动画继续按逻辑坐标运行；保持世界姿态时还会换算矩形尺寸，使单位切换不改变其世界空间大小。场景数据记录单位策略，保存与克隆后继续从当前父级解析。

布局准备只绑定真正需要原生矩形的控件。静态兄弟控件不因另一个控件播放动画而被整批转换。成员变化时更新参与者缓存；稳定帧复用缓存，预览释放临时矩形后再更新相关记录。

## 布局中的动画

在容器控制的子元素上，动画应修改布局输入，例如 `UILayoutElement.preferred_width` 与 `preferred_height`。布局会继续控制最终矩形与位置，并同步移动受影响的兄弟元素。

```python
from infernux.ui import UILayoutElement
from infernux_tween import Ease, twn_preferred_size

layout = image.game_object.add_component(UILayoutElement)
layout.preferred_width = 80
layout.preferred_height = 32
twn_preferred_size(layout, (160, 48), 0.4).set_ease(Ease.OUT_CUBIC)
```

独立摆放的元素可以使用锚点、尺寸和位置动画。直接动画一个受容器控制的输出属性时，下一次布局仍会根据布局输入计算该属性。

## 编辑器预览

矩形预览分别保存创作数据与临时运行状态。关闭预览或执行 Undo 时恢复原有布局驱动、尺寸、位置和共享 UI 绑定；不把预览期间创建的父矩形留在场景里。同一矩形及其临时父矩形不能同时由互不关联的预览持有。

## 验证范围

Tween 插件仓库包含矩形驱动、序列化、预览与 Undo、屏幕和世界布局的功能检查。真实 GPU 检查覆盖自动 Flow、布局输入动画、退出布局后的像素与点击位置，并核对 UI 子树的保存数据。原生矩形姿态与驱动数学还分别在 Windows、Linux、Web 和 Android 上运行检查。

这些检查不代表全部 Player、物理接触、复杂布局组合和平台集成都已完成验收。
