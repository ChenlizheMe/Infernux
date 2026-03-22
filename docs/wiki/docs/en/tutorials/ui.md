# UI System — Buttons, Text, and Why Your Health Bar Matters

<div class="class-info">
Tutorial &nbsp;|&nbsp; <a href="../../zh/tutorials/ui.html">中文</a>
</div>

## Overview

InfEngine's runtime UI system is screen-space, component-based, and feels familiar if you've used Unity UI. You build interfaces from **UICanvas** → **UIText / UIImage / UIButton** components, style them with properties, and respond to events with Python callbacks. No HTML, no CSS, no web browser embedded in your game. Just GPU quads and text.

## Architecture

```
UICanvas          ← root container, defines screen space
├── UIImage       ← background panel
├── UIText        ← "Score: 42"
├── UIButton      ← clickable, fires events
│   └── UIText    ← button label
└── UIImage       ← health bar fill
```

Every UI element lives on a GameObject with a UI component. The `UICanvas` is the root — it sets the reference resolution and render mode.

## Quick Start: Hello UI

### 1. Create a Canvas

In the editor: **Right-click Hierarchy → UI → Canvas**. This creates a UICanvas with a reference resolution.

Or in code:

```python
from InfEngine import *
from InfEngine.ui import *

class SetupUI(InfComponent):
    def start(self):
        # Canvas is usually set up in the editor,
        # but here's how to do it in code
        canvas_go = GameObject("MainCanvas")
        canvas = canvas_go.add_py_component(UICanvas)
        canvas.sort_order = 0
```

### 2. Add Text

```python
class ScoreDisplay(InfComponent):
    score: int = 0
    
    def start(self):
        self.text = self.game_object.get_py_component(UIText)
        self.text.text = "Score: 0"
        self.text.font_size = 32
        self.text.color = vector4(1, 1, 1, 1)  # white
    
    def add_score(self, points):
        self.score += points
        self.text.text = f"Score: {self.score}"
```

### 3. Add a Button

```python
class MenuButton(InfComponent):
    def start(self):
        btn = self.game_object.get_py_component(UIButton)
        btn.on_click.add_listener(self.on_button_click)
    
    def on_button_click(self):
        Debug.log("Button clicked! Time to do something awesome.")
```

## UI Components Reference

### UICanvas

The root container for all UI elements. Every UI hierarchy needs exactly one.

| Property | Description |
|----------|-------------|
| `render_mode` | `RenderMode.ScreenSpaceOverlay` (default) or others |
| `sort_order` | Draw order — higher values render on top |
| `reference_resolution` | The resolution you design for (scales automatically) |

### UIText

Renders text on screen. No font files needed — the engine bundles a default font.

```python
text = self.game_object.get_py_component(UIText)
text.text = "Game Over"
text.font_size = 48
text.color = vector4(1, 0, 0, 1)       # Red
text.alignment_h = TextAlignH.Center
text.alignment_v = TextAlignV.Middle
```

### UIImage

Displays a texture or solid color rectangle. Great for backgrounds, health bars, icons.

```python
img = self.game_object.get_py_component(UIImage)
img.color = vector4(0.2, 0.2, 0.2, 0.8)   # Semi-transparent dark
img.raycast_target = False                   # Don't block clicks
```

### UIButton

A clickable element. Inherits from `UISelectable`, so it has hover/press/disabled states with color transitions.

```python
btn = self.game_object.get_py_component(UIButton)
btn.interactable = True
btn.normal_color = vector4(0.3, 0.3, 0.3, 1)
btn.highlighted_color = vector4(0.5, 0.5, 0.5, 1)
btn.pressed_color = vector4(0.2, 0.2, 0.2, 1)
btn.on_click.add_listener(self.handle_click)
```

## Common Patterns

### Health Bar

```python
class HealthBar(InfComponent):
    max_health: float = serialized_field(default=100.0)
    _current: float = 100.0
    
    def start(self):
        self.fill = self.game_object.get_py_component(UIImage)
    
    def set_health(self, value):
        self._current = Mathf.clamp(value, 0, self.max_health)
        ratio = self._current / self.max_health
        # Scale the fill image width
        # Green when full, red when low
        r = Mathf.lerp(1.0, 0.0, ratio)
        g = Mathf.lerp(0.0, 1.0, ratio)
        self.fill.color = vector4(r, g, 0, 1)
```

### Fade In/Out Screen

```python
class ScreenFade(InfComponent):
    """Full-screen overlay for scene transitions."""
    
    def start(self):
        self.img = self.game_object.get_py_component(UIImage)
        self.img.color = vector4(0, 0, 0, 0)  # Start transparent
        self.img.raycast_target = True          # Block input during fade
    
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

### Simple Dialog Box

```python
class DialogBox(InfComponent):
    _visible: bool = False
    
    def start(self):
        self.panel = self.game_object  # The dialog panel
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

## Event System

UI events flow through the `UIEventSystem`. When the user clicks or hovers, events are dispatched to the topmost `raycast_target` element.

The `PointerEventData` object contains:
- `position` — screen coordinates of the pointer
- `button` — which mouse button (Left, Right, Middle)
- `click_count` — single/double/triple click

## Tips

1. **Always put UI under a Canvas.** UI components without a Canvas parent won't render.
2. **Set `raycast_target = False`** on decorative images. Otherwise they'll eat your clicks.
3. **Use `sort_order`** to layer multiple canvases (e.g., HUD at 0, popup at 10, debug at 100).
4. **Reference resolution** — design at 1920×1080, the engine scales to actual screen size.

## See Also

- [UICanvas API](../api/UICanvas.md)
- [UIText API](../api/UIText.md)
- [UIButton API](../api/UIButton.md)
- [UIImage API](../api/UIImage.md)
- [Coroutines Tutorial](coroutines.md) — for UI animations
