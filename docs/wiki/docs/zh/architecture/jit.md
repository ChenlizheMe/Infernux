---
category: 架构
tags: ["jit", "性能", "python", "numba", "并行"]
date: "2026-04-16"
---

# 深入理解：Infernux 的 JIT 加速脚本系统

Infernux 是一个把整个玩法层都写在 Python 里的游戏引擎。这是有意为之的架构选择——Python 带来了热重载、庞大的生态和极低的上手门槛——但它同时也意味着每一帧的更新循环默认都是解释执行的。当你需要在一帧内触碰十万个 Transform 时，解释执行的速度远远不够。这篇文档会完整地走一遍引擎 JIT 子系统是怎么解决这个问题的，它在底层做了哪些设计决策，以及这项工作接下来要往哪里走。

完整的评估方法和性能表格请参阅技术报告原文：[*Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting*（arXiv:2604.10263）](https://arxiv.org/pdf/2604.10263)。

---

## 1 &ensp; 问题的起点：一千次边界穿越的代价

假设一个最朴素的 Python 脚本要逐个更新场景中每个实体的位置：

```python
for obj in scene.objects:
    pos = obj.transform.position          # Python → C++ → Python
    pos.y += math.sin(time + pos.x)       # 纯 Python 数学运算
    obj.transform.position = pos          # Python → C++ → Python
```

每一次迭代都要穿越 pybind11 边界两次：读一次、写一次。每次穿越都要拿 GIL、做 C++ 与 Python 之间的类型转换、走一遍 pybind11 的 type-caster 调度。10 个对象的时候这根本看不见。10 万个对象的时候，花在边界开销上的时间反而超过了实际数学运算本身。

所以引擎的性能故事分两章，必须按顺序读。

---

## 2 &ensp; 第一章：批量数据桥

在任何 JIT 编译能帮上忙之前，我们必须先消除逐对象的边界穿越。批量数据桥（batch bridge）通过一次调用把引擎状态搬进连续的 NumPy 数组来实现这一点：

```python
positions = batch_read(targets, "position")   # 一次穿越，返回 (N,3) 数组
jit_wave_kernel(positions, time_value, count)  # 纯数组运算，零穿越
batch_write(targets, positions, "position")    # 一次穿越，写回
```

`batch_read` 内部走一个分层快速路径：

1. **C++ Transform 路径** —— 对 `position`、`rotation`、`scale` 等 Transform 属性，直接调用 C++ 侧的 `TransformECSStore::GatherVec3` / `GatherQuat`，在释放 GIL 的情况下把 SoA 存储 memcpy 进预分配好的 NumPy buffer。
2. **ComponentDataStore 路径** —— 对用户自定义的 `InxComponent` 字段（Float64、Int64、Bool、Vec2–Vec4 等类型），走 C++ 的类型化批量 gather/scatter。
3. **Python 回退** —— 对以上都不覆盖的情况，用 `getattr` 循环逐个收集。这是慢路径，但永远正确。

`create_batch_handle()` 还能进一步缓存底层的 C++ `Transform*` 指针，使得后续帧不需要再做 O(N) 的 pybind11 目标列表转换。

结果：边界穿越成本从 O(N)（每帧每属性）降到 O(1)。到达内核的数据是密排的、缓存友好的 NumPy ndarray——正是 LLVM 向量化器最喜欢的输入形态。

---

## 3 &ensp; 第二章：编译内核

边界穿越消除之后，剩下的成本就是 Python 解释器执行内层循环的开销。在 CPython 里对 10 万个浮点数做一个 sin-and-add 循环要花几毫秒；编译成原生 SIMD 代码后只要几十微秒。这就是 JIT 要做的事。

### 3.1 &ensp; `njit` 装饰器

引擎只暴露一个入口：`from Infernux.jit import njit`。根据调用方式不同，行为也不同：

| 用法 | 行为 |
|---|---|
| `@njit` | 编译一个串行版本，通过字节码键缓存。|
| `@njit(cache=True, auto_parallel=True)` | 编译**两个**版本（串行 + 并行），对并行版本做 AST 重写，并构建一个自动选择的调度器。|

当机器上没有安装 Numba 时，`njit` 退化成一个透明的透传装饰器：原始 Python 函数照常执行，同时发出运行时警告，让性能退化是可见的而非静默的。

### 3.2 &ensp; 基于字节码的编译缓存

传统 Numba 缓存以 `(module, function_name)` 为键，这在游戏引擎里会出问题——脚本在一次编辑会话中可能被热重载几十次，模块身份每次都变，但代码本身可能没变。

Infernux 的缓存键是这样的：

```
(co_filename, func_name, sha256(co_code)[:16], kwargs_tag)
```

对原始字节码做 `sha256` 意味着：如果函数体没有变，即便热重载销毁并重建了模块对象，缓存的机器码依然可以复用。如果字节码变了，只重编译那一个函数，其余内核继续使用已有缓存。

这存储在一个进程内的 `dict`（`_compiled_cache`）里。没有文件系统级别的锁竞争，没有需要手动清理的 `.nbi` 文件，也不存在 Numba 内置文件缓存和 Nuitka 的冻结导入器交叉干扰的边角情况。

### 3.3 &ensp; 优雅退化与自愈

JIT 子系统预设了三种运行时环境：

- **Numba 完整可用** —— 正常编译并缓存。
- **Numba 缺失** —— 回退到纯 Python。`ensure_jit_runtime()` 可选地通过 pip 自动安装 Numba（一条自愈路径：import 失败 → 安装 → 重新 import → 成功）。
- **Nuitka 打包后** —— Numba 的文件缓存被禁用（因为 `co_filename` 不再指向真实的 `.py` 文件），但内存中的编译仍然有效。这让打包构建拥有 JIT 性能，而不需要终端用户安装编译器工具链。

每个编译后的函数对象都携带一个 `.py` 属性指向原始 Python 可调用对象，所以内省和调试总是有回退依据。

---

## 4 &ensp; 自动并行化：从 `range` 到 `prange`

这是 JIT 子系统中最具 Infernux 自身特色的部分，值得展开来讲。

### 4.1 &ensp; 动机

Numba 支持 `prange`——一种并行的 range，把循环迭代分配到多个线程上执行。但要求每个玩法程序员手动写 `prange`、自行验证没有数据竞争、在并行不划算时手动回退串行版本，这在实际项目里是不现实的。引擎应该在安全的前提下自动做这件事。

### 4.2 &ensp; AST 重写器

当设置了 `auto_parallel=True` 时，引擎取出被装饰函数的源码，解析成 Python AST，然后运行 `_AutoParallelRangeTransformer`——一个 `ast.NodeTransformer` 子类。它遍历每个 `for` 节点，依次问四个问题：

1. **这是一个计数循环吗？** —— 迭代器必须是直接的 `range(...)` 调用。迭代器风格的循环（`for x in arr`）和 `while` 循环永远不会被改动。

2. **循环体里没有不支持的控制流吗？** —— `return`、`break`、`yield`、`try/except` 出现在循环体内时，直接放弃重写。这些结构要么并行语义不清晰，要么 Numba 的并行后端不支持。

3. **写入模式安全吗？** —— 重写器检查两类模式：
   - **支持的归约操作**：`total += arr[i]`、`product *= arr[i]` 之类的增量赋值，被识别为可交换的归约，Numba 可以在线程间分割。
   - **尴尬并行的索引写入**：形如 `out[i] = expr` 的数组写入，其中索引就是循环变量。因为每次迭代写入唯一的索引，不存在数据竞争。

   如果两类模式都没匹配到，循环保持串行。

4. **执行重写**：如果所有检查都通过，AST 中的 `range` 被替换为 `prange`，修改后的源码通过 `exec` 编译，得到的函数对象交给 Numba 以 `parallel=True` 进行编译。

```
原始：   for i in range(n):  out[i] = arr[i] * scale
                                  ↓ AST 重写
重写后： for i in prange(n): out[i] = arr[i] * scale
```

### 4.3 &ensp; 为什么语法层面的分析就够了

一个真正的编译器需要完整的别名分析才能证明 `out[i]` 和 `arr[j]` 在 `i ≠ j` 时不会别名。引擎的重写器故意跳过这一步，使用一个语法启发式：如果写入目标是 `<数组>[<循环变量>]`，就假设是不相交的。

这对游戏引擎的主要使用场景是成立的：按实体 ID 索引的 SoA 属性更新。`positions[i]`、`velocities[i]`、`scales[i]`——引擎自带的每一个内核都是这个模式。启发式拒绝任何更复杂的东西，这是保守安全的方向：你可能错过一个并行化机会，但永远不会引入数据竞争。

### 4.4 &ensp; 预构建的附属文件

对打包构建来说，启动时做实时 AST 重写有额外延迟。构建系统可以通过 `build_auto_parallel_sidecar_source()` 预先生成 `.autop.pyc` 附属文件。运行时，`_load_prebuilt_auto_parallel_variant()` 优先查找附属文件，只有没找到时才回退到实时重写。

---

## 5 &ensp; 双变体调度器

并行执行并不总是更快。N 小的时候，线程池开销占主导；N 大的时候，并行吞吐才能胜出。引擎用一个双变体调度架构来处理这个问题。

### 5.1 &ensp; 编译

设置 `auto_parallel=True` 时，引擎从同一份源码编译出**两个** Numba 函数：

- `fn_serial` —— 以 `parallel=False` 编译，保持原始 `range` 循环。
- `fn_parallel` —— 以 `parallel=True` 编译，`range` 被重写为 `prange`。

两者在字节码缓存中由各自的 `kwargs_tag` 区分。

### 5.2 &ensp; 预热与锁定

`warmup()` 函数做的不只是触发编译。对双变体内核，它跑一个**两阶段基准测试**：

1. 分别调用 `fn_serial(*args)` 和 `fn_parallel(*args)` 一次来触发编译（冷启动结果丢弃）。
2. 测量两个变体在实际预热参数上的稳态延迟。
3. 把更快的那个**锁定**为默认调度目标。

锁定发生在场景加载时。逻辑是：如果你的场景只有 50 个对象，串行路径可能更快，调度器不应该每帧都付线程池开销。如果场景有 50 万个对象，并行路径胜出，调度器应该从第一个真实帧就使用它。

### 5.3 &ensp; 运行时回退

即使锁定之后，并行变体仍可能在运行时失败（比如 Numba 的线程池碰到操作系统限制）。调度器捕获异常并在本次会话剩余时间里**自愈**到串行变体，同时输出一条警告日志。

---

## 6 &ensp; 预热和热重载的实际体验

### 6.1 &ensp; 冷启动成本

通过 Numba/LLVM 编译一对串行+并行变体，在现代桌面 CPU 上每个函数通常需要 50–200 ms。这在场景加载时可以接受，但在游戏运行中途不能接受。

### 6.2 &ensp; 预热辅助

```python
from Infernux.jit import warmup

def on_scene_load():
    dummy_positions = np.zeros((1000, 3), dtype=np.float32)
    warmup(jit_wave_kernel, dummy_positions, 0.0, 1000)
```

通过在场景加载（或加载画面）时调用 `warmup()`，编译成本对玩家来说是隐藏的。dummy 参数同时也作为双变体锁定的基准测试输入。

### 6.3 &ensp; 热重载的集成

当一个脚本在编辑器里被修改并重载时：

1. 模块重新导入，产生新的函数对象。
2. `njit` 为每个函数计算字节码键。
3. 如果键匹配到已有缓存条目 → 复用，零编译成本。
4. 如果键是新的（代码变了）→ 只重编译那一个函数。其他内核不受影响。

这意味着编辑一个脚本函数最多触发一次 50–200 ms 的重编译，而不是全项目重建。对一个编辑器优先的引擎来说，这和原始吞吐一样重要。

---

## 7 &ensp; 性能结果

技术报告里最干净的测量是*纯计算基准*——没有渲染，不写回 Transform，只比较脚本吞吐本身。

| 元素数 | Auto-parallel JIT | NumPy（无 JIT） | Unity IL2CPP |
|---|---|---|---|
| 10k | >3000 FPS | ~800 FPS | ~2400 FPS |
| 100k | ~2200 FPS | ~120 FPS | ~600 FPS |
| 1M | **848 FPS** | ~80 FPS | ~123 FPS |

在 1M 元素下：**比 Unity IL2CPP 快 6.9 倍**，**比非 JIT NumPy 路径快 10.5 倍**。

这里真正有意义的不是某一个数字，而是一条缩放曲线：随着实体数增长，JIT 路径优雅退化，而解释执行路径直接崩溃。当内层循环被编译、边界穿越被批量化之后，Python 仍然可以胜任实时 authoring 场景。

---

## 8 &ensp; 当前局限

这个系统对自己不能解决的问题是诚实的：

- **边界现在成了瓶颈。** 每次 `batch_read` / `batch_write` 调用仍然穿越 pybind11、拿 GIL、做类型转换、再返回。内核运算不再是天花板了，边界延迟才是。
- **静态变体锁定假设了稳定的工作负载。** 预热基准测试在场景加载时做一次测量就锁定了串行或并行。如果运行时工作负载发生剧烈变化，锁定的选择可能变得不再最优。
- **语法层面的别名分析是保守的。** 一些实际上可以安全并行化的循环会因为写入模式不匹配 `arr[i] = ...` 模板而被拒绝。

---

## 9 &ensp; 接下来的方向

### 9.1 &ensp; 无锁命令环

下一个计划中的边界优化是在 Python 脚本线程和 C++ 运行时之间建立一个无锁环形缓冲区。把 batch dispatch 从同步的 pybind11 调用改成命令包入队，由原生侧在没有 GIL 竞争的情况下消费。这直接攻击的是 JIT 路径暴露出来的边界穿越瓶颈。

### 9.2 &ensp; 自适应 JIT：动态问题规模监控

这是我们目前最积极探索的方向，它触及了游戏工作负载的一个本质矛盾。

**核心观察：** 游戏中的问题规模不是静态的。玩家这一秒可能面对 20 只怪兽，下一秒马上就要面对 300 只。一个粒子系统在平静场景里可能发射 500 个粒子，爆炸时瞬间变成 50,000 个。最优的并行化策略取决于 N——而 N 每一帧都在变。

静态变体锁定（第 5.2 节所述）在预热时测一次就选定串行或并行，然后一直用下去。这作为默认策略是合理的，但在动态场景下它把性能留在了桌上：

- 如果锁定了并行，而 N 降到了交叉点以下，线程池开销就在浪费周期。
- 如果锁定了串行，而 N 突然飙升，内核就错过了跨核扩展的机会。

**正在探索的思路**是一个*自适应 JIT 调度器*：一个轻量级监控器运行在一个独立的子线程上，持续观察每个算子实际被调用时的问题规模（N）。当它检测到当前变体对于观测到的 N 不再最优时，异步地触发变体替换：

```
主线程：     kernel(data, N=30)   →  串行变体（小 N 时快）
                 ...
             kernel(data, N=8000) →  串行变体（已不最优！）
                 ...
监控线程：   观察到 N 持续上升 → 发信号"切换到并行"
                 ...
主线程：     kernel(data, N=12000) → 并行变体（现在是最优的）
```

关键的设计约束有三个：

- **热路径上零竞争。** 主线程绝不能因为等待监控器的决策而阻塞。变体替换是*最终一致*的——主线程读一个原子指针来决定调用哪个变体，监控器异步地更新这个指针。
- **滞回以避免抖动。** 切换变体有成本（缓存行失效、线程池唤醒）。监控器应该用窗口化平均或者指数平滑来区分持续的规模变化和瞬时尖峰。
- **逐算子粒度。** 不同内核的串行-并行交叉点不同。监控器为每个注册的算子独立追踪 N。

这将使 JIT 子系统不仅仅是一个静态编译器，而是一个运行时自适应的执行层——一个根据游戏实际产生的工作负载，逐帧持续调优并行化策略的系统。

这项工作还处于早期探索状态，但它瞄准的是一个真实的空缺：现有科学计算领域的 JIT 系统假设问题规模是稳定的，而游戏的工作负载天生就是动态的。弥合这个差距，是引擎性能故事的下一个前沿。

---

## 延伸阅读

- [Infernux: A Python-Native Game Engine with JIT-Accelerated Scripting（arXiv:2604.10263）](https://arxiv.org/pdf/2604.10263) —— 完整技术报告，包含基准测试方法论和评估。
- [架构概述](about.md) —— C++ 运行时、pybind11 绑定层和 Python 生产层如何组合在一起。
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