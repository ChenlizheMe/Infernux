# statistics

<div class="class-info">
函数位于 <b>Infernux.jit</b>
</div>

```python
statistics(fn: Callable[..., Any]) → Statistics
```

## 描述

Detached compilation/decision/mapped-memory snapshot, without executing fn.

<!-- USER CONTENT START --> description
返回由 `jit.compile` 生成函数的独立快照，包括当前选择的模式、准备耗时、
特化、缓存决策，以及后端支持时的原生代码内存占用。
<!-- USER CONTENT END -->

## 参数

| 名称 | 类型 | 描述 |
|------|------|------|
| fn | `Callable[..., Any]` |  |

## 示例

<!-- USER CONTENT START --> example
```python
from infernux import jit

report = jit.statistics(integrate)
print(report.selected_mode, report.specializations)
```
<!-- USER CONTENT END -->
