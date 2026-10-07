# compile

<div class="class-info">
函数位于 <b>Infernux.jit</b>
</div>

```python
compile(fn: Callable[..., Any] = ...) → Any
```

## 描述

Compile a CPU function; automatic serial/parallel selection is enabled by default.

<!-- USER CONTENT START --> description
此装饰器用于 CPU 数值计算。Infernux 会分析函数并选择串行实现或已证明
安全的原生并行实现。桌面目标缺少 CPU 编译器时会直接拒绝声明；Web 则在
Cook 阶段删除装饰器并直接生成普通 Python 字节码，Player 运行时不再执行
这个装饰器。这属于构建目标合同，而不是发生错误后悄悄切换后端；未 Cook
的 JIT 调用在缺少编译器时仍会明确报错。
<!-- USER CONTENT END -->

## 参数

| 名称 | 类型 | 描述 |
|------|------|------|
| fn | `Callable[..., Any]` |  (default: `...`) |

## 示例

<!-- USER CONTENT START --> example
```python
from infernux import jit

@jit.compile
def integrate(positions, velocities, dt):
    for i in range(len(positions)):
        positions[i] += velocities[i] * dt
```
<!-- USER CONTENT END -->
