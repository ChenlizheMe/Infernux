# warmup

<div class="class-info">
函数位于 <b>Infernux.jit</b>
</div>

```python
warmup(fn: Callable[..., Any]) → None
```

## 描述

Prepare a compiled CPU function on isolated inputs; errors propagate.

<!-- USER CONTENT START --> description
使用隔离的参数副本编译并验证一种运行时签名。准备失败会直接抛出，
不会改用另一个后端再次执行。无 CPU JIT 的目标会在 Cook 普通 Python
字节码时删除该调用；运行时 API 本身不会变成 no-op。
<!-- USER CONTENT END -->

## 参数

| 名称 | 类型 | 描述 |
|------|------|------|
| fn | `Callable[..., Any]` |  |

## 示例

<!-- USER CONTENT START --> example
```python
jit.warmup(integrate, positions, velocities, 1.0 / 60.0)
```
<!-- USER CONTENT END -->
