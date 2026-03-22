# FullScreenEffect

<div class="class-info">
类位于 <b>InfEngine.renderstack</b>
</div>

**继承自:** [RenderPass](RenderPass.md)

## 描述

全屏后处理效果基类。自定义后处理从这里继承。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `FullScreenEffect.__init__(enabled: bool = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| requires | `ClassVar[Set[str]]` |  *(只读)* |
| modifies | `ClassVar[Set[str]]` |  *(只读)* |
| menu_path | `ClassVar[str]` |  *(只读)* |
| requires_per_frame_rebuild | `ClassVar[bool]` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `setup_passes(graph: RenderGraph, bus: ResourceBus) → None` | Override to add fullscreen passes to the render graph. |
| `get_shader_list() → List[str]` | Return shader paths required by this effect. |
| `inject(graph: RenderGraph, bus: ResourceBus) → None` | Inject this effect into the render graph. |
| `get_params_dict() → Dict[str, Any]` | Get serializable parameters as a dictionary. |
| `set_params_dict(params: Dict[str, Any]) → None` | Restore parameters from a dictionary. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 运算符

| 方法 | 返回值 |
|------|------|
| `__repr__() → str` | `str` |

<!-- USER CONTENT START --> operators

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for FullScreenEffect
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
