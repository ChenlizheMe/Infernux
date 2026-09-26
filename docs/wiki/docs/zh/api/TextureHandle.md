# TextureHandle

<div class="class-info">
类位于 <b>Infernux.rendergraph</b>
</div>

## 描述

渲染图中声明或导入的纹理资源句柄。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `TextureHandle.__init__(name: str, format: Format, is_camera_target: bool = ..., size: Optional[Tuple[int, int]] = ..., size_divisor: int = ..., samples: int = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` |  |
| format | `Format` |  |
| is_camera_target | `bool` |  |
| size | `Optional[Tuple[int, int]]` |  |
| size_divisor | `int` |  |
| samples | `int` |  |
| asset_guid | `str` | 导入纹理资产的 GUID；渲染图临时纹理和 RenderTexture 附件为空字符串。 |
| depth | `int` | 纹理深度；2D 纹理为 `1`。 |
| is_volume | `bool` | 该句柄是否引用 3D 纹理。 |
| is_depth | `bool` | Returns True if this texture uses a depth format. *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 运算符

| 方法 | 返回值 |
|------|------|
| `__repr__() → str` | `str` |
| `__eq__(other: object) → bool` | `bool` |
| `__hash__() → int` | `int` |

<!-- USER CONTENT START --> operators

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example
> **示例状态：** 当前尚未为此符号验证 0.4.0 示例。请以上方签名为准；不要根据其他引擎中的同名 API 推测行为。
<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
