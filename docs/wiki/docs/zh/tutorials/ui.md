# UI 系统 — 按钮、文字，以及你的血条为什么很重要

<div class="class-info">
教程 &nbsp;|&nbsp; <a href="../../en/tutorials/ui.html">English</a>
</div>

## 概述

InfEngine 的运行时 UI 系统是屏幕空间的、基于组件的，用过 Unity UI 的人会觉得很亲切。你用 **UICanvas** → **UIText / UIImage / UIButton** 组件来搭建界面，用属性设置样式，用 Python 回调响应事件。没有 HTML，没有 CSS，没有在游戏里内嵌浏览器。就是 GPU 四边形和文字渲染。

## 架构

```
UICanvas          ← 根容器，定义屏幕空间
├── UIImage       ← 背景面板
├── UIText        ← "分数: 42"
├── UIButton      ← 可点击，触发事件
│   └── UIText    ← 按钮文字
└── UIImage       ← 血条填充
```

每个 UI 元素都在一个带 UI 组件的 GameObject 上。`UICanvas` 是根 —— 它设定参考分辨率和渲染模式。

## 快速开始：Hello UI

### 1. 创建画布

在编辑器中：**右键 Hierarchy → UI → Canvas**。会创建一个带参考分辨率的 UICanvas。

或者用代码：

```python
from InfEngine import *
from InfEngine.ui import *

class SetupUI(InfComponent):
    def start(self):
        canvas_go = GameObject("MainCanvas")
        canvas = canvas_go.add_py_component(UICanvas)
        canvas.sort_order = 0
```

### 2. 添加文本

```python
class ScoreDisplay(InfComponent):
    score: int = 0
    
    def start(self):
        self.text = self.game_object.get_py_component(UIText)
        self.text.text = "分数: 0"
        self.text.font_size = 32
        self.text.color = vector4(1, 1, 1, 1)  # 白色
    
    def add_score(self, points):
        self.score += points
        self.text.text = f"分数: {self.score}"
```

### 3. 添加按钮

```python
class MenuButton(InfComponent):
    def start(self):
        btn = self.game_object.get_py_component(UIButton)
        btn.on_click.add_listener(self.on_button_click)
    
    def on_button_click(self):
        Debug.log("按钮被点击了！是时候做点厉害的事了。")
```

## UI 组件参考

### UICanvas

所有 UI 元素的根容器。每个 UI 层级需要且只需要一个。

| 属性 | 描述 |
|------|------|
| `render_mode` | `RenderMode.ScreenSpaceOverlay`（默认） |
| `sort_order` | 绘制顺序 — 数值越大越在上面 |
| `reference_resolution` | 设计分辨率（会自动缩放适配） |

### UIText

在屏幕上渲染文字。不需要字体文件 —— 引擎内置了默认字体。

```python
text = self.game_object.get_py_component(UIText)
text.text = "游戏结束"
text.font_size = 48
text.color = vector4(1, 0, 0, 1)       # 红色
text.alignment_h = TextAlignH.Center
text.alignment_v = TextAlignV.Middle
```

### UIImage

显示纹理或纯色矩形。适合做背景、血条、图标。

```python
img = self.game_object.get_py_component(UIImage)
img.color = vector4(0.2, 0.2, 0.2, 0.8)   # 半透明深色
img.raycast_target = False                   # 不拦截点击
```

### UIButton

可点击的元素。继承自 `UISelectable`，有悬停/按下/禁用状态和颜色过渡效果。

```python
btn = self.game_object.get_py_component(UIButton)
btn.interactable = True
btn.normal_color = vector4(0.3, 0.3, 0.3, 1)
btn.highlighted_color = vector4(0.5, 0.5, 0.5, 1)
btn.pressed_color = vector4(0.2, 0.2, 0.2, 1)
btn.on_click.add_listener(self.handle_click)
```

## 常用模式

### 血条

```python
class HealthBar(InfComponent):
    max_health: float = serialized_field(default=100.0)
    _current: float = 100.0
    
    def start(self):
        self.fill = self.game_object.get_py_component(UIImage)
    
    def set_health(self, value):
        self._current = Mathf.clamp(value, 0, self.max_health)
        ratio = self._current / self.max_health
        r = Mathf.lerp(1.0, 0.0, ratio)
        g = Mathf.lerp(0.0, 1.0, ratio)
        self.fill.color = vector4(r, g, 0, 1)
```

### 全屏淡入淡出

```python
class ScreenFade(InfComponent):
    """场景过渡用的全屏遮罩。"""
    
    def start(self):
        self.img = self.game_object.get_py_component(UIImage)
        self.img.color = vector4(0, 0, 0, 0)  # 开始时透明
        self.img.raycast_target = True          # 淡入淡出时拦截输入
    
    def fade_to_black(self):
        self.start_coroutine(self._fade(1.0))
    
    def fade_from_black(self):
        self.start_coroutine(self._fade(0.0))
    
    def _fade(self, target_alpha):
        current = self.img.color.w
        while abs(current - target_alpha) > 0.01:
            current = Mathf.move_towards(current, target_alpha, Time.delta_time)
            self.img.color = vector4(0, 0, 0, current)
            yield WaitForEndOfFrame()
```

### 简单对话框

```python
class DialogBox(InfComponent):
    _visible: bool = False
    
    def start(self):
        self.panel = self.game_object
        self.text = self.game_object.find_child("DialogText").get_py_component(UIText)
        self.btn = self.game_object.find_child("OKButton").get_py_component(UIButton)
        self.btn.on_click.add_listener(self.dismiss)
        self.panel.active = False
    
    def show(self, message):
        self.text.text = message
        self.panel.active = True
    
    def dismiss(self):
        self.panel.active = False
```

## 事件系统

UI 事件通过 `UIEventSystem` 分发。当用户点击或悬停时，事件会传递给最顶层的 `raycast_target` 元素。

`PointerEventData` 对象包含：
- `position` — 指针的屏幕坐标
- `button` — 哪个鼠标按钮（左、右、中）
- `click_count` — 单击/双击/三连击

## 小贴士

1. **UI 必须在 Canvas 下。** 没有 Canvas 父级的 UI 组件不会渲染。
2. **装饰性图片设置 `raycast_target = False`。** 否则它们会吞掉你的点击事件。
3. **用 `sort_order` 分层。** 比如 HUD 设 0，弹窗设 10，调试面板设 100。
4. **参考分辨率** — 按 1920×1080 设计，引擎会自动缩放到实际屏幕尺寸。

## 另请参阅

- [UICanvas API](../api/UICanvas.md)
- [UIText API](../api/UIText.md)
- [UIButton API](../api/UIButton.md)
- [UIImage API](../api/UIImage.md)
- [协程教程](coroutines.md) — 做 UI 动画
