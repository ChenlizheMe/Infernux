# InfEngine Wiki

Welcome to the InfEngine documentation wiki.

## Quick Links

- [API Reference (English)](en/api/index.md)
- [API 参考手册 (中文)](zh/api/index.md)

## Tutorials

New to InfEngine? Start here:

| Tutorial | Description |
|----------|-------------|
| [Physics](en/tutorials/physics.md) | Colliders, Rigidbody, triggers, raycasting, character controller |
| [Audio](en/tutorials/audio.md) | AudioSource, 3D spatial sound, music, footsteps |
| [UI](en/tutorials/ui.md) | Canvas, Text, Image, Button, health bars, screen fade |
| [Coroutines & Time](en/tutorials/coroutines.md) | WaitForSeconds, time control, async patterns |
| [Rendering](en/tutorials/rendering.md) | Render pipeline, post-processing, custom effects |
| [Building](en/tutorials/building.md) | Standalone game export with Nuitka, Hub launcher |

中文教程：[物理](zh/tutorials/physics.md) · [音频](zh/tutorials/audio.md) · [UI](zh/tutorials/ui.md) · [协程](zh/tutorials/coroutines.md) · [渲染](zh/tutorials/rendering.md) · [构建](zh/tutorials/building.md)

## Getting Started

InfEngine is an open-source game engine with a C++17/Vulkan backend and Python scripting frontend. Write game logic in Python while the engine handles high-performance rendering, physics, and audio.

### Hello World

```python
from InfEngine import *

class HelloWorld(InfComponent):
    speed: float = serialized_field(default=5.0)
    
    def start(self):
        Debug.log("Hello, InfEngine!")
    
    def update(self):
        self.transform.rotate(vector3(0, self.speed * Time.delta_time, 0))
```

## Modules

| Module | Description |
|--------|-------------|
| [InfEngine](en/api/index.md) | Core types — GameObject, Transform, Scene, Component |
| [InfEngine.components](en/api/InfComponent.md) | Component system — InfComponent, serialized_field, decorators |
| [InfEngine.core](en/api/Material.md) | Assets — Material, Texture, Shader, AudioClip |
| [InfEngine.coroutine](en/api/Coroutine.md) | Coroutines — WaitForSeconds, WaitUntil, WaitWhile |
| [InfEngine.input](en/api/Input.md) | Input system — keyboard, mouse, touch |
| [InfEngine.math](en/api/vector3.md) | Math — vector2, vector3, vector4, quaternion |
| [InfEngine.mathf](en/api/Mathf.md) | Math utilities — clamp, lerp, smooth_step |
| [InfEngine.physics](en/api/Physics.md) | Physics — Rigidbody, colliders, raycasting |
| [InfEngine.rendergraph](en/api/RenderGraph.md) | Render graph — textures, passes, formats |
| [InfEngine.renderstack](en/api/RenderStack.md) | Render stack — pipelines, post-processing effects |
| [InfEngine.scene](en/api/SceneManager.md) | Scene management |
| [InfEngine.timing](en/api/Time.md) | Time — delta_time, time_scale, frame timing |
| [InfEngine.ui](en/api/UICanvas.md) | UI — Canvas, Text, Image, Button |
| [InfEngine.debug](en/api/Debug.md) | Logging and diagnostics |
| [InfEngine.gizmos](en/api/Gizmos.md) | Visual debugging aids |
