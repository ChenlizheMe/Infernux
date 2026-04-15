---
category: 架构
tags: ["jit", "性能", "python"]
date: "2026-04-16"
---

# JIT 加速脚本

这一页把技术报告 [*Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting*](https://arxiv.org/pdf/2604.10263) 里最关键的 JIT 部分整理成一个更适合项目文档阅读的版本。

## 为什么需要 JIT 路径

Python 适合做玩法、工具和热重载工作流，但如果每帧都对大量对象做逐个属性访问，Python 和 C++ 之间的边界开销会迅速压垮吞吐。Infernux 的做法不是放弃 Python，而是把最热的内层循环挪到更接近原生代码的执行路径上。

报告里把这件事拆成两个协同机制：

- 一个 batch bridge，用一次边界穿越把引擎状态搬进连续的 NumPy 数组。
- 一个基于 Numba 的可选 JIT 路径，把标注过的 Python 更新函数编译成 LLVM 机器码。

重点不是把 Python 换成别的脚本语言，而是继续让 Python 负责 authoring，同时尽可能削掉高频更新里的解释器成本。

## 先 batch，再 JIT

报告里强调，JIT 不是孤立存在的，它依赖 batch 数据桥的配合。典型每帧更新流程大致如下：

```python
positions = batch_read(targets, "position")
jit_wave_kernel(positions, time_value, count)
batch_write(targets, positions, "position")
```

这样做的意义是把 Python 与 C++ 之间的调用次数固定在每帧常数级，而不是随着对象数量线性增长。JIT 内核则只面对紧密排列的数组，这也是 Numba 最擅长的输入形态。

## Infernux 在 Numba 之上补了什么

项目里的 JIT 装饰器并不只是简单调用一次 `numba.njit`。报告总结的增强点主要有三条：

- 当机器上没有安装 Numba 时，自动退回纯 Python 路径，同时发出运行时警告，让性能退化是可见的。
- 使用基于字节码的编译缓存，使缓存能跨模块重载和编辑器热重载继续生效。
- 对 Nuitka 打包提供部分兼容。Numba 本身仍然依赖运行时的 CPython，所以这不是完全意义上的 AOT 替代，更像是分发层面的兼容处理。

换句话说，JIT 被整合进了引擎自己的重载、缓存和打包模型，而不是单纯做成一个独立优化开关。

## 自动并行化

JIT 部分里最有 Infernux 自身特色的是 auto-parallel 模式。开启之后，AST 重写器会扫描形如 `for i in range(n)` 的计数循环，并在循环体满足一组保守条件时把它提升成 `prange`，交给 Numba 做线程级并行。

它不会尝试并行化所有循环，以下几类都会被拒绝：

- `for x in arr` 这类迭代器循环。
- `while` 循环。
- 包含 `break`、`yield`、异常处理、提前返回等不受支持控制流的循环体。
- 写入模式看起来不是“按索引写入彼此不重叠数组位置”的循环。

报告也明确说明，这里的别名分析只是语法层面的检查，不是完整编译器意义上的 alias analysis。但对 Infernux 主要场景来说，这已经够用：最常见的内核就是按实体 ID 索引批量写 position、rotation、scale 等 SoA 列。

## 预热与热重载

报告指出，串行版和并行版一起编译会带来冷启动开销，在测试机器上通常每个被装饰函数大约是 50 到 200 ms。Infernux 通过两种方式把这个成本移出交互主路径：

- 提供 warm-up helper，在场景加载时就预编译已注册的 JIT 内核。
- 在热重载场景里，只重编译发生改动的函数，未改动函数继续复用已有机器码缓存。

这其实比单纯“跑得快”更重要，因为编辑器工作流要求的不只是峰值吞吐，还要求脚本热改时延迟可控。

## 报告里的核心结果

JIT 部分最干净的实验是纯计算测试，也就是不涉及渲染、也不把结果写回 Transform，只比较脚本吞吐本身。在这个实验里：

- auto-parallel JIT 路径在 10k 到 1M 元素范围内都维持了很高的帧率。
- 在 1M 元素规模下，报告中的 runtime 结果达到 848 FPS。
- 同一配置下，它相对 Unity IL2CPP 参考实现达到 6.9 倍吞吐，相对非 JIT 的 NumPy 路径达到 10.5 倍。

真正有意义的不是某个单点数字，而是这说明：当边界调用被 batch 化、热循环被编译后，Python 仍然可以支撑实时 authoring 场景，而不必退回到“编辑器一套语言、运行时另一套语言”的双轨结构。

## 当前边界与下一步

报告最后也很坦白：剩下最大的瓶颈已经不是 JIT 内部的算术吞吐，而是 Python 与 C++ 的通信边界本身。每一次 batch dispatch 仍然要走 pybind11、拿 GIL、做类型转换、再返回。

所以 JIT 之后的下一步路线图不是继续卷编译器，而是改造边界层，例如 lock-free command ring。JIT 已经把“算得慢”的问题压下去了，接下来限制上限的是“过边界还不够便宜”。

## 完整技术报告

完整的系统背景、实验设置和性能表格请直接阅读原报告：

- [Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting（arXiv:2604.10263）](https://arxiv.org/pdf/2604.10263)