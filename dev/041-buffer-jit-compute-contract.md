# 041：引擎 Buffer、CPU JIT 与 GPU Compute 执行合同

> 2026-09-13 追加：已删除活动路径中的旧设备传输/命令实现、device API 构建目标和 LLVM 分派器；137 个实际编译单元的依赖记录只保留架构/能力描述，无旧设备或 LLVM 运行时头文件。最终 88/88 原生、126 项 Compute/CPU JIT 定向测试通过；编译器子仓库的私有安装目录 CI 已修正并在本机实测。源码外部的未用后端、共享 IR、正式 ABI 与多平台交付仍未完成，因此 B05 和总勾选数不变。详细证据见执行记录。

> 2026-09-13 复核：本附录 **37/76**，主计划 **104/257**，合计 **141/333（42.3%）**，剩余 **192 项**。GPU 编译上下文直接使用 SPIR-V 编译器，产物只含 SPIR-V/元数据；移除单实现继承层、TIC 输出/重复序列化及相关哈希路径，保留引擎 .inxgpu 缓存。Compute 资源持有同一 Device/Queue 服务，修复宿主 wrapper 先释放导致的悬空与退出顺序；新增独立上下文、IR/layout 和生命周期合同回归。最终原生 **88/88**（16 项 Vulkan）、Python **6199 通过 / 11 跳过**；更新 MCP 后的 041Lab 雪面、早期软体和 RenderTexture 可见消费回归通过，正常退出日志干净。完整原生依赖/类型工厂、编译服务/ABI、刚软体数量阶梯性能和多平台交付仍未收口，B05 不勾选。下方早期段落仅作历史证据。

> 2026-09-15 Player 验证：Windows source-less Player 已实际运行 GPUJelly，`fixed_update` 累计 **1.12s**，GPU 批处理约 **2.18ms**；修复了内置 GPU kernel 源码元数据和 `.pyc` vendor loader。`test_compute.py`、`test_game_builder.py`、生命周期调度和 Player service graph 合计 **358 passed / 1 skipped**。这些是新增验收证据，尚未把跨平台、DataAsset 和旧 Taichi 迁移条目标为完成。

> 2026-09-15 回归修复：Python 组件热替换现在继承原组件的 `enabled` 状态；PlayerBootstrap 对轻量宿主缺失的可选启动属性使用确定默认值；路径架构仍统一经 `path_utils`。组件替换、PlayerBootstrap、构建设置和路径架构专项合计 **61 passed**，全量回归已推进至 **4327 passed / 6 skipped** 后进入下一处独立问题。

修订：2026-09-09，依据最新用户讨论。状态：实施已恢复。Windows 主仓已能构建并审计内置 GPU JIT wheel 载荷；`inx.buffer` 已完成 Windows Vulkan 全量/区间 set/get、按需 staging、有序异步上传和同一引擎共享 ComputeHost。公开的 `@inx.compute.kernel`、`inx.compute.index`、`inx.compute.launch` 已贯通无设备编译、SPIR-V 元数据和 Infernux RHI 执行，标量、vector3、多维/间接索引与多 Buffer 真实 Vulkan 数值回归通过；引擎 lifecycle 自动记录一阶段内的上传/launch，并把参数更新和 kernel tasks 合入尽可能少的 queue submission。Jolt 刚体状态/BoxCollider SoA 可由 C++ 一次异步提交直写多块 GPU buffer；线性/角冲量两块 GPU buffer 可一次回读等待后直接反馈 Jolt，不经过 Python/NumPy。编译仍暂时借用上游 `Program`/ndarray 注解作为私有 lowering 壳；Program、field/SNode/ndarray 源码闭包及原生运行时尚未完全删除。`@inx.jit.compile` 已成为 CPU 新入口并可直接原地消费 CPU buffer。动态 Mesh 发布已删除逐帧全内容哈希，改用资产 GUID 代际或运行时对象代际；Jolt 已有空间宽相刚体候选查询。GPU Mesh 直连、刚软体完整交互、Linux、Player 和编译器深裁仍未完成。

最新确认：本期只交付 JIT，AOT 不进入项目构建或 Player；引擎 wheel 与适用的 Player 一起携带 CPU/GPU 两套 JIT 工具。Taichi fork 更名 taichi_for_infernux，目录为 external/taichi_for_infernux。

本文是 [041 主计划](041-runner-long-foundation-plan.md) A04/A05 的执行附录。它取代统一 `compute.hpc(device=...)`、强制 NumPy 边界和公开 `resident/upload/download/dispatch` 方案，不是并行维护的新备选 API。

## 1. 已确认的作者模型

| 入口 | 职责 | 执行模型 |
|---|---|---|
| `inx.buffer(...)` | 引擎拥有的类型化连续数据与资源视图 | 显式 CPU/GPU 存储；供计算、网格、渲染等消费 |
| `buffer.set_data(data)` / `buffer.get_data()` | 写入/取回数据 | GPU 对应上传/回读；不隐式双向同步 |
| `inx.vector2 / vector3 / vector4` | 现有小写向量入口 | 复用现有原生数值类型，不新增 vec3 别名或另一套向量对象 |
| `@inx.jit.compile` | CPU 函数优化 | 普通函数调用；整个函数描述算法；自动选择合法串行/并行 |
| `@inx.compute.kernel` + `inx.compute.launch(kernel, params=...)` | GPU 计算 | 函数描述一个 work item；由引擎提交 GPU 工作 |
| `inx.compute.index(buffer)` | kernel 的执行域与当前元素索引 | 编译期识别所引用 buffer 的逻辑元素域，不从参数列表猜 dim |

CPU/GPU 不再共用执行语义；入口已经表达计算设备，buffer 的 device 表达存储位置。GPU kernel 不自动回退为 CPU 函数；需要 CPU 实现的产品必须提供并明确选择对应算法，不能将已提交任务失败重放。历史设备兼容回退条款已被本轮拆分替代。

NumPy 可选互操作继续保留，尤其已有 Mesh/CPU JIT 调用；不再要求全部输入输出先转换为 NumPy。`compute.nn` 不在本轮实现神经网络，也不以旧占位符冻结未来 NN 命名或数据面；Torch/DLPack 不成为本期依赖。

## 2. B01：inx.buffer 数据与传输合同

2026-09-09 实施基线：新增 CPU/GPU `Buffer`、标量/vector2/3/4 布局、显式全量/区间 `set_data/get_data`、复用输出、fill/zero、CPU 索引和禁止 GPU 隐式索引；GPU 由引擎 ComputeHost 创建 DeviceLocal Storage|Vertex buffer，并用按需 Upload/Readback staging 经引擎 ComputeQueue 往返。`set_data` 提交后不再立即等待，后续 kernel/readback 由同队列顺序连接；同一上传 staging 被 CPU 再写前才等待上一使用，`get_data` 仍是明确同步边界。原生描述冻结 float32/int32/uint32、逻辑元素数、1—4 lane 和元素步长；真实 Windows Vulkan 12×vector3 全量及区间往返通过。跨消费者的共享 staging 优化与更广资源互操作归入 B02，不作为 B01 作者数据合同的隐藏前置条件。

2026-09-12 描述补齐：公开不可变 `BufferDescription`，对当前 dense 标量/向量布局给出 shape、标量类型、lane 属性偏移、元素步长、容量、有效字节范围、device、usage、ownership、Buffer 代际和不泄漏 Vulkan 句柄的 RHI 资源身份/代际。真实 Vulkan 分配验证身份稳定且可用；CPU 资源不伪造 GPU 身份。`inx.buffer` 继续被 value codec 明确拒绝为可序列化值，只能由 GUID 资产或作者显式数据重新构造。

2026-09-12 借用与退休补齐：`Buffer.view(offset, count)` 形成无复制的一维逻辑范围，拥有独立 wrapper 代际和有效字节范围，同时复用同一 CPU allocation 或 RHI resource identity。CPU 重叠视图互相可见；GPU kernel 通过正式参数 ABI 消费视图 byte offset，实测只修改目标范围。关闭 owner wrapper 后 view 仍持有存储；`MeshRenderer` 绑定 resident vertex buffer 后由引擎共享持有 native allocation，解除绑定再沿统一资源退休。kernel bind-group 缓存只弱引用非在途资源，submission ticket 未完成时才强持有实际 buffer；完成后按正常帧边界回收过期代际。真实 Vulkan 连续以 7/19/41/83 容量重建并关闭资源，缓存恢复到稳定基线，没有随代际增长。多维 typed view 仍是后续布局扩展，不影响当前 dense 一维借用合同。

2026-09-12 B01 收口：只读 view 公开在资源描述中，CPU `set_data`/索引写和 GPU 编译元数据中的 write/read-write 在提交前统一拒绝；只读资源作为纯 read kernel 参数正常执行。零尺寸 shape 明确报错，不为 Vulkan 制造零字节占位 allocation。至此 B01 的公开数据、传输、驻留、借用和退休合同完成；异步 readback、跨 Mesh/实例/粒子/物理的更多互操作及 staging 性能属于 B02。

- [x] 引擎提供一个原生资源描述：元素类型、逻辑 shape、元素步长/属性偏移、字节范围、device、访问用途、所有权和资源代际。复用 RHI BufferResource/Handle，不以 Taichi field/SNode 作为所有者。
- [x] 第一阶段交付连续 dense buffer，数值标量和现有 vector2/3/4；精确冻结整数宽度、浮点精度、溢出/转换和错误行为。后续矩阵/结构体沿权威类型布局扩展，不临时为果冻发明私有结构体。
- [x] `shape=N, dtype=inx.vector3` 表示 N 个三维向量，不是 3N 个执行元素。vector3 的 native/NumPy/storage/vertex 布局不可假设天然同 stride；只维护一套权威布局描述，必要转换显式或在 GPU 上完成。
- [x] `set_data` 接受兼容的 CPU buffer、向量/数值序列以及可选 NumPy；只传输指定有效区间，布局不匹配在提交前报告，不隐式补齐、截断或猜测 dtype。跨 GPU 资源复制另外明确，不把设备 buffer 当成主机指针。
- [x] `get_data()` 返回已可读取的 CPU 类型化 buffer，不逐元素预先生成 Python 对象；支持向复用的目标 CPU buffer 取数据，具体 out/区间参数在实现前统一。默认同步回读会等待，应在文档明确，不能叫异步。
- [x] CPU buffer 允许普通 Python 索引；GPU buffer 的元素索引用于 kernel。普通 Python 不通过一次 `gpu[i]` 隐式触发回读，必须调用 get_data。kernel 元素是数值而非 Python/pybind 对象；局部向量修改与数组写回的值语义明确。
- [x] 初始化提供 fill/zero 等批量路径；新分配未初始化数据的规则必须明确，测试不依赖显存碰巧为零。容量固定或显式重建，不在 launch 时猜测增长。
- [x] GPU 数据跨 kernel/物理步/帧驻留；set_data 后未再上传的 CPU 修改不自动可见；get_data 得到快照，不是持续同步镜像。默认不长期保存一份无用途的 CPU 副本。
- [x] NumPy 转换声明 alias/copy、dtype、shape、可写性和生命周期；普通 CPU NumPy 到独立 GPU 不是零拷贝。CPU buffer 暴露受控连续存储供 JIT，不能靠 Python 逐元素包装维持性能。
- [x] 自建 buffer、资源借用视图、共享 Mesh 数据的 ownership 清楚；关闭、owner 销毁、容量/布局变化及在途 GPU 引用复用引擎退休机制。不能以插件卸载销毁引擎拥有的存储。
- [x] buffer 是运行期数据资源，不自动变成 serialized_field 可持久化资产。保存/重新构造沿 GUID 和显式数据规则，不能序列化裸 GPU 地址。

验收：标量/vector3、空 buffer、非整工作组长度、部分更新、相同/重叠视图、只读及非法布局、复用输出、跨帧驻留、在途释放、资源代际变化。只在必要边界做布局/能力检查，不扫描数据内容或逐帧计算 hash。

## 3. B02：引擎内部资源互操作与 GPU 化

源码基线：RhiBuffer.h 已有 BufferResource；RhiComputeHost 提供宿主服务；粒子已有状态/实例/可见性/indirect buffer；MeshRenderer 的相关编辑接口仍使用 std::vector<Vertex>；GizmosDrawCallBuffer 有独立 CPU 几何数组。以下是接入工作，不将已有 Handle 当作完整公开数据面。

| 消费者 | 041 工作与验收 | 不应混淆的边界 |
|---|---|---|
| 动态 Mesh | 位置/法线/索引视图、计算写入后直接绘制、区间更新和 bounds 发布 | 修改共享内置 Sphere 前显式实例化；不自动重建 MeshCollider |
| 法线/切线/bounds 工具 | 在数据所在地运行；GPU 路径避免全网格回读 | CPU 裁剪若需要 bounds，使用明确保守 bounds 或有时序的少量回读 |
| 材质/实例数据 | buffer 绑定、布局验证、实例变换、可见索引、indirect 消费 | 合适 usage/layout 是直连前提，不承诺任意两资源都零拷贝 |
| 粒子 | 现有 buffer 接同一资源/访问描述；复用已有 GPU 计算和生命周期 | 不为统一名字重写已工作的全部粒子算法或强制改成 Python kernel |
| Gizmo | 批量几何/实例数据接入；测量 CPU 提交与 GPU 绘制瓶颈 | 不假定所有卡顿来自 CPU 计算，不因换容器宣布优化完成 |
| 物理交换 | 碰撞体批量快照、宽相候选和反馈冲量的数据通路 | 先测宽相/传输收益；刚体 solver 不因这次计划自动整体迁 GPU |
| CPU JIT | 直接读取 CPU buffer 的 native 数值布局，保留 NumPy 兼容 | 不自动下载 GPU buffer，也不在循环里调用 pybind getter |

- [ ] 为每个 GPU 化候选记录现有耗时、规模、传输量、依赖和收益；优先动态 Mesh/法线/实例/物理交换这条真实消费者链。其它引擎机制先审计，只有有实测收益和明确语义才迁移。
- [x] 计算写 → 渲染读、计算写 → 下一计算读、CPU 写 → GPU 读、GPU 写 → CPU 读共用引擎任务依赖/barrier/退休；保留资源而非每步全设备 idle。（2026-09-12：buffer 携带精确最后写入票据；常驻 Mesh 图形提交只等待其实际写入而非全计算队列最新提交；批内 compute barrier、上传和同步/异步回读共用 ComputeQueue。）
- [x] 异步 readback 使用独立明确的完成合同及现有任务基础；不得改变同步 get_data 语义，不能用上一帧冲量伪装同一固定步的双向接触。（2026-09-12：`get_data_async` 返回精确提交的 `Readback`，`done` 只轮询票据，`get_data` 完成并返回 CPU Buffer；每次请求独占 staging，源关闭后仍由在途所有权保活。）
- [ ] Texture/RenderTexture 保留图像格式、采样与 mip 语义，只统一底层资源生命周期；不强制把纹理伪装成 buffer。
- [x] 专项统计 Python/native 边界次数、分配/map 次数、上传/回读字节数、CPU 提交、GPU 时间、等待和退休积压；不开默认逐帧日志，不做内容 hash。（2026-09-12：`compute.statistics()` 提供累计边界/dispatch/submission、传输批次与字节、staging 分配、host map、CPU submit/wait 时间及当前在途提交；GPU timestamp 由显式 `set_profiling_enabled` 启用，切换是唯一 drain 边界。）

2026-09-09 物理交换切片：Jolt broadphase AABB 查询已避免远处刚体进入 Python 候选；BoxCollider 已能按候选一次输出扁平 SoA 与 body 索引，支持复用超额容量的 CPU/GPU buffer。刚体运动状态快照同样允许写入超额容量；GPU 输出由 C++ 收集后用一次 transfer submission 写入七块刚体状态或六块 Box 状态 buffer，不生成 Python/NumPy 中间数组。反馈端把线性/角冲量两块 GPU buffer 放入一次 copy submission、一次 queue wait 后直接施加到 Jolt。真实 Windows Vulkan→Jolt 测试验证质量 2/4 的刚体获得正确线速度，状态和 Box body index 的反向 GPU 上传也通过。尚未把软体窄相/约束、GPU Mesh 和这两个边界接成完整固定步，因此不能据此声称双向接触、数量矩阵或 300 FPS 通过。

2026-09-12 精确跨队列依赖、异步回读与统计：`ComputeBuffer` 现在记录上传或 kernel 写入的最后一个真实 submission ticket；Vulkan 计算队列按仍在途的具体 ticket 返回 timeline/value。DrawFrame 从当前常驻顶点资源集合选择最新实际写入，因此后续无关 compute submission 不再扩大 Compute→Graphics 等待。帧快照公开被消费的写 serial、后台最新 serial 和是否仍需等待，真实 Vulkan Mesh 回归故意在每次变形后追加独立无关 kernel，并证明二者 serial 分离且渲染保持正常。公开 `Readback` 为每次请求拥有独立 staging 和源资源租约；同步 `get_data` 未改变。公开累计统计覆盖 Python/native 边界、dispatch/submission、传输批次/字节、staging/map、CPU submit/wait、在途提交和按需 GPU timestamp，不开默认日志。CPU 48 项和 GPU kernel/Mesh/物理定向回归通过。B02 仍缺候选性能基线与 Texture 边界复核，不能勾总项。

## 4. B03：GPU kernel 的单 work-item 编译语义

2026-09-09 执行底座：新增引擎私有多 dispatch 提交，一组已编译 kernel 在同一个 RHI command buffer 中顺序执行，kernel 间由 compute write → compute read/write barrier 连接，整组只做一次 ComputeQueue submit；稳定 kernel/buffer 组合复用 bind group，组内禁止同一 kernel 偷换绑定。真实 Windows Vulkan 用同一 257 元素 buffer 连续执行两次缩放并只在最终 `get_data` 回读，结果正确。该证据仅覆盖执行/同步，不代表 `kernel/index/launch` 作者编译语义已完成。

2026-09-09 编译器直连实证：编译 fork 为动态外部数组生成的任务曾携带固定 `advisory_total_num_threads=131072`，这不能作为普通作者填写的 dim，也不能直接决定小 buffer 的派发规模。引擎 RHI 已支持显式稀疏 binding 和 uniform/storage 类型；测试用实际 `inx.buffer` 逻辑长度与编译器工作组宽度推导 group 数，将同一 Taichi SPIR-V 的 arguments uniform buffer 与数据 storage buffer 直接绑定后得到正确 128 项结果。执行没有调用 `Program.launch_kernel`、Taichi ndarray 分配或 Taichi command list。最终 `index(buffer)` 前端仍须把权威执行域写入编译产物，不能长期从任意“第一个数组”猜测。

2026-09-09 公开前端纵向切片：`index(buffer)` 已由 Infernux AST 前端确认为唯一一维执行域，运行期 group 数只由该 Buffer 的实际元素数和编译工作组宽度推导；普通调用方不填 dim。公开 kernel 使用引擎 Buffer 和标量，内部 dummy NumPy 只为尚待替换的上游类型 lowering 服务，不成为作者资源。真实 wheel 隔离安装后，257 项 int32 两次参数化原地变换，以及 129 项 vector3 位置/速度双 Buffer 积分均通过；编译临界区结束后没有顶层 `taichi` 模块残留。当前尚未覆盖原子、辅助函数、共享存储、复杂控制流和跨 launch 的公开批处理，因此 B03 不勾选完成。

2026-09-09 调度与语法增量：每个组件 lifecycle phase 现在自动开启 compute recording，作者连续调用 `launch` 不再逐次提交；显式 get_data 或 GPU→Jolt 反馈才结束当前依赖段。RHI kernel 可为同一 pipeline 保留多组资源绑定，使同一批中相同 kernel 使用不同参数 uniform；update-only batch 也有明确 transfer→consumer barrier。1000 次 64×64 独立 Vulkan 微基准从立即提交的约 34.9/38.0 微秒每 launch 降至批量记录的约 19.0/17.7 微秒。另以一维 order buffer 驱动二维 float buffer 的间接行索引、局部串行 for 和分支已通过真实 SPIR-V 执行，说明 XPBD 所需的基础多维访问可迁移；原子、辅助函数和最终求解器仍未验证。

2026-09-12 B03 公开合同收口：真实 Windows Vulkan 回归故意把 17 项执行域放在第二个参数、把 41 项输出放在第一个参数，只有输出前 17 项被修改，证明 group 数只来自显式 `index(domain)`，不会按参数顺序或最大 Buffer 猜测。`index` 声明前的标量语句、其后的普通 `for` 与 `while` 均逐 work item 串行执行；257 项非工作组整数倍继续覆盖尾部范围掩码。编译前拒绝逐线程 Python `return`，CPU Buffer 传入 GPU kernel 直接要求显式传输，不进入作者函数或 CPU 重放。公开 `launch` 仍只原地写 Buffer，不产生隐式回读。新增显式 `@inx.compute.function` 数值辅助函数和 `inx.compute.atomic_add`，真实 SPIR-V 分别验证辅助调用和 17 work-item 的 int32 归约；任意 Python 闭包与 kernel/helper 递归在编译前给出确定错误。041 当前消费者的跨 work-item 归约可用原子或独立 reduce launch；不公开尚无消费者的 workgroup shared/barrier，也不让用户填写 group size。公开错误分为 `ComputeCapabilityError`、`KernelCompilationError` 与 `ComputeExecutionError`，分类边界只翻译一次异常，不重试、不降级。`infernux.compute_kernel_gpu` 真实 Vulkan 回归与 `test_compute.py` 32 项通过。B03 作者执行语义完成；B05 私有编译器深裁和各平台发行仍是独立未完成项。

- [x] 一个 kernel 整体是单 work item，包括 index 声明之前的语句。`index(buffer)` 是执行域声明及 ID 值，不是执行到该行再开线程。
- [x] 第一阶段明确一维逻辑域：shape=N 的元素缓冲支持 vector 元素；调用方不填 dim、group 数。二维/三维域、多个不同 shape 的联合域先定义后开放，不从首个参数或最大长度猜测。
- [x] 普通 for/while 保持当前 work item 内串行语义，不沿用 Taichi 顶层 for 自动并行。引擎生成 dispatch/group 划分和必要范围掩码；循环展开是编译优化，不产生新 work item。
- [x] 多次全局阶段分为多个 launch 并建立依赖；不同 work item 无顺序保证。共享存储、原子和 barrier 的最低集合按真实消费者定义，不能拿工作组 barrier 充当全 GPU barrier。（041 最低集合为全局 `atomic_add` + 跨 launch 的 RHI compute barrier；当前消费者不需要、公共面不暴露 workgroup shared/barrier。）
- [x] 若支持 workgroup barrier/shared memory，尾部非活动线程仍正确参与 barrier，只屏蔽非法内存访问；不得统一早退造成死锁。高级 group-size/indirect 参数不成为普通 launch 的必填项。（041 不支持 workgroup shared/barrier；非整组尾部已通过 257 项真实回归，作者不填写 group size。）
- [x] GPU kernel 通过 buffer 原地输出；launch 不隐式收集每线程 Python 返回值，不自动回读。需要标量归约时写明确结果 buffer 再取回；CPU JIT 返回语义单独定义。
- [x] 支持数值向量运算、辅助函数和确定范围的原子/归约；编译前诊断不支持的动态对象/闭包/递归。别名分析保留必要约束，但不承诺静态证明所有间接索引无竞争。（vector3、显式 `compute.function`、全局 `atomic_add`、独立 reduce launch 均有真实 Vulkan 证据；参数只允许 Buffer/标量，闭包和递归前置拒绝。）
- [x] 设备能力、编译合法性与执行失败分别报告；GPU kernel 不作为 CPU jit 函数重放。GPU 资源与 CPU 入参不匹配必须显式传输。

示例（待实现，不作为已运行证据）：

```python
positions = inx.buffer(shape=1024, dtype=inx.vector3, device="gpu")
velocities = inx.buffer(shape=1024, dtype=inx.vector3, device="gpu")
positions.fill(inx.vector3(0, 10, 0))
velocities.fill(inx.vector3(0, 0, 0))

@inx.compute.kernel
def integrate(positions, velocities, dt):
    i = inx.compute.index(positions)
    v = velocities[i]
    v.y -= 9.81 * dt
    velocities[i] = v
    positions[i] = positions[i] + v * dt

inx.compute.launch(integrate, params=(positions, velocities, 1.0 / 60.0))
cpu_positions = positions.get_data()  # 明确回读并等待，仅 CPU 需要时执行。
cpu_positions[0] = inx.vector3(0, 20, 0)
positions.set_data(cpu_positions)
```

## 5. B04：Infernux JIT 新装饰器与 V8 级分析能力

2026-09-09 执行底座：`@inx.jit.compile` 已作为 CPU 作者入口落地，默认启用现有 typed-HIR 的合法串行/并行分析、有限决策缓存、warmup 与执行错误不重放规则；CPU `inx.buffer` 直接借用其连续 NumPy 存储进入 JIT，返回同一底层数组时恢复为原 buffer，GPU buffer 明确拒绝并要求走 `compute.launch`。旧 `njit` 暂留迁移兼容。JIT/HIR/runtime/compute 定向回归 122 项通过。低开销原生拆箱、分层编译、类型反馈增强、代码内存预算及 V8 对照尚未完成，因此 B04 不勾选。

2026-09-12 CPU 作者数据合同：`@inx.jit.compile` 在进入后端前只接受数值标量、非 object NumPy、CPU `inx.buffer` 和 Infernux vector2/3/4；返回值限定为数值标量、NumPy、传入的 CPU Buffer、引擎向量、由这些值组成的 tuple 或 `None`。未知 Python 对象与 object array 不进入 object mode，也不在失败后执行 Python。实际被已编译函数引用的同模块普通数值辅助函数由编译边界递归登记，无需第二套 helper 装饰器；分支、局部循环和归约由真实 nopython 测试覆盖。JIT/HIR/runtime/compute 定向回归 `122 passed`。该证据冻结第一阶段数据/控制流表面；优化层级、类型反馈和 V8 对照仍未完成。

2026-09-12 CPU 选择与准备边界复核：现有 typed HIR 先排除循环携带依赖、未知副作用及不安全别名，只有合法候选才生成并行版本；运行签名还会以实际 NumPy overlap 把调用固定到串行。静态工作量或隔离 warmup 决定串/并行，未证明并行是合法编译决策而非执行失败 fallback；真实调用一旦选择目标只执行一次，异常不从头重放。warmup 克隆保持数组/view/别名关系并比较返回与写入，只作用于隔离输入。041 的函数调用边界足以覆盖当前消费者，因此明确不做函数内 OSR/去优化；后续若出现必须长时间运行且需要中途换版本的消费者，再另立状态映射合同。相关 122 项回归通过。

目标不是只给 Numba 装饰器改名，而是建设引擎拥有的、低调用成本、具备类型反馈与优化决策的 CPU 函数执行系统。V8 作为编译延迟、分层优化、类型特化与依赖失效的参考；不是引入 JavaScript VM，也不把 V8 说成自动多核循环引擎。

- [x] `@inx.jit.compile` 正常调用，函数描述完整算法；继续利用 Numba/llvmlite 的 CPU codegen，不恢复 Numba GPU，不为改名无理由重写机器码后端。
- [x] 冻结第一阶段支持的标量、CPU buffer、NumPy、vector 值、辅助函数、分支/循环/归约子集；返回标量、CPU buffer、NumPy 或 None 的生命周期和分配规则明确。未知 Python 对象/引擎副作用不通过 object mode 或 no-op 悄悄运行。
- [ ] 审计现有 Typed HIR，建立/复用 CFG、SSA 或等效数据流、活跃性、类型/范围传播、读写与别名分析；给每个优化声明前提和正确性测试，不增加重复的影子 IR。
- [ ] 类型反馈与有界单态/多态特化：dtype/rank/layout/alias、已注册类型/字段布局、函数依赖代际参与必要签名。普通数量变化优先作为运行参数，不为每个 N 重新编译；决策缓存和机器码内存分别有界。
- [ ] 优化向量/标量拆箱、固定字段偏移访问、常量传播、死代码消除、循环不变量外提、受控内联、可证明安全的边界检查消除和 SIMD。复用后端已有 pass，增加编译报告证明实际消除了 Python 对象访问，而非重复做一套 pass 列表。
- [x] CPU 串并行选择先保证合法性，再依据成本/规模与已有预热测量；有循环携带依赖、未知别名、顺序副作用时使用合法串行实现。无法证明可并行不是错误后的 fallback，而是编译决策。
- [ ] 分层优化：准备阶段可用的低编译成本基线与热点优化版本共用数值语义；复用 Numba 可行 pass 配置并测量代价，不假设它天然具备 V8 全套 tier/OSR。优化版本在安全调用边界发布；真实有副作用的调用只执行一次。
- [ ] 保留自动串并行与跟踪，补齐编译延迟/特化/优化层级/决策原因/缓存命中/代码占用。热点计数与采样有界，稳定签名快速调用，不每帧重走 AST 分析/设备探测/源码 hash。
- [x] 准备/预热明确，不在首次关键交互静默编译数秒；只对可隔离输入做串并行试跑，不因 profiling 重复发射引擎副作用。收益不足时保留合法基线，不堆叠反复试错。
- [ ] 必要 guard 检查入口类型/布局/依赖代际，在执行前选定有效特化；类型/辅助函数/字段 schema 更新精确失效。已发生写入后的异常不能从头重放到 Python/串行版本。
- [ ] 与 A01 权威 schema 结合：只优化有受控布局/生命周期的字段访问，不能缓存任意 Component 内存偏移跨发布代际使用。041 负责 authored 热重载失效，042 复用到动态 owner。
- [x] 评估函数内去优化/OSR 的真实必要性和成本；本期默认只在函数调用/发布边界切换。若必须支持执行中去优化，先定义状态映射和副作用连续性并单独确认范围，不能以 fire-forced 为由删除正确性保护。（041 当前消费者无需函数中途切换，明确不实现 OSR；真实调用在边界选定一个目标且异常不重放。）
- [ ] 制作 CPython、当前 JIT、修订 JIT、V8/Node 与原生参考对照：小函数高频调用、分支密集、向量操作、循环携带依赖、可并行循环、归约、签名变化、辅助函数热更新。锁定版本/机器/精度/数据/线程数，区分冷编译、预热、稳态、P95 和内存；V8 单线程比较与多核吞吐分开报告。
- [ ] “超越 V8”是目标，必须逐工作负载给出数据与不占优原因；不能用单一数值循环胜出宣称全面超越，也不把降低精度/省略工作算优化。性能阈值在基线采集后冻结，不将写入此计划当成已达到 V8 水平。

```python
@inx.jit.compile
def integrate_cpu(positions, velocities, dt):
    for i in range(len(positions)):
        v = velocities[i]
        v.y -= 9.81 * dt
        velocities[i] = v
        positions[i] = positions[i] + v * dt

# CPU buffer 或合法 NumPy 输入；正常调用，无 launch，无 GPU 自动选择。
integrate_cpu(cpu_positions, cpu_velocities, 1.0 / 60.0)
```

## 6. B05：内置 wheel 的纯 Vulkan 计算编译模块

用户已确认：裁剪后的 taichi_for_infernux 是引擎本体 wheel 的组成部分，不再作为项目插件或单独的 Taichi wheel/.inxpkg 分发。编辑器安装引擎即具备对应平台的编译载荷，无需插件安装/启用，也不在首次调用联网补包。源码仍作为有明确上游来源的 external 依赖维护，不散入引擎业务代码。

### B05.1 编译器输入输出边界

2026-09-13 辅助函数续进：引擎的 `compute.function` 继续按现有语义内联，删除上游独立 `real_func` 特化/调用、对应缓存与 Program 函数登记，并移除不再使用的原生导出。必要的共享优化 IR 保留。借用 ASTBuilder/配置对象通过绑定所有权保持原生上下文存活，不添加代理登记或轮询校验；带类型注解的多返回值先构造可转换序列，再按声明类型处理，避免直接修改 tuple。嵌套辅助函数、默认参数、分支/循环、类型化返回及失败生命周期均有正式回归。完整 Windows 原生/Python 回归及现有 Lab 消费场景通过；这仍不替代最终 ABI、依赖删除映射和发行矩阵验收。

2026-09-13 所有权续进：编译内核及其原生 IR 改由单次请求持有，Python/Program 不再长期登记全部内核。原生对象先完整构造并交给 Python，再执行 AST 转换；保留编译异常时，其栈中的内核会保持上下文存活，丢弃后正常释放。临时生成源码及请求复制的全局变量在转换完成后解除引用，不依赖编辑器下一次 GC。无设备成功/失败交替编译和真实 Vulkan 定向回归均通过；这解决的是内核请求生命周期，尚不代表全部原生 Function/类型工厂/遗留源码或完整编译服务已收口。

当前纵向切片已导出每个任务的 SPIR-V、线程组宽度以及具名资源元数据（`arguments` / `external_buffer`、`uniform` / `storage`、参数索引），并由引擎 RHI 直接消费。这里刻意不用 Taichi 内部整数枚举作为跨层 ABI，也不让 RHI 将全部绑定误当 storage。临时测试仍借用上游 Python AST 前端和 Program 取得已 lower 的 kernel；因此证明的是执行/runtime 可替换，不是纯编译入口已经完成。

2026-09-13 Python 闭包现状：正式安装33个私有模块，导入全部限定于 Infernux 内部，不再扫描/置换公共 `taichi.*`。Matrix/Vector/Struct 的 field 工厂、MatrixField/StructField、FieldsBuilder/root、SNode host access、runtime materialize/sync 和动态 SNode AST 兜底已从调用链与源码移除；`_compiler_types.py`、`runtime_ops.py`、`mesh.py`、`snode.py` 以及旧梯度 AST 检查器已删除，而非只依赖安装排除。编译现在直接消费引擎提供的 BufferType 描述和标量类型，不再构造 NumPy 占位数组；单个入口不再自动创建梯度内核，类方法栈帧探测及类包装器已移除。矩阵/向量表达式、必要的双域数值函数和 IR 保留；无设备编译已接入正式 CTest，真实 Vulkan kernel/helper/atomic/vector-buffer 与数值读写回归通过。原生 Program/IR 所有权、遗留 AOT/其它后端源码、编译 ABI 和完整发行矩阵仍待处理，不能据此视为完整 B05 收口。

无设备编译切片已增加默认构造的 compiler-only 上下文：独立进程在没有 Infernux 图形引擎、VkDevice、队列、runtime materialization 或 ndarray 分配的条件下，将带外部连续数组参数的 kernel 编译为 SPIR-V。显式传 `None` 不会被当作设备自动探测。这个切片仍使用 `Program` 类型名和上游 ndarray 类型注解来承载 AST，下一步由 Infernux `kernel/index` 前端和独立 compiler service 取代，不能把它当作最终公开 API。

2026-09-11 旧设备适配器已从主链和源码中删除：原生绑定不再导出 `_EngineComputeDevice` / `_attach_engine_compute_host`，编译器子构建不再接收引擎 include 或链接 ComputeHost，`taichi/runtime/infernux` 的 device/stream/buffer/pipeline 及其独立适配测试也已移除。无设备生成 SPIR-V 后仍由 Infernux 自有 RHI/Vulkan 队列执行，真实 Windows Vulkan kernel 回归通过。当前剩余问题是进一步拆除私有 lowering 对 `Program`、field/SNode/ndarray Python 导入闭包和 gfx runtime 链接对象的依赖，不再存在第二条引擎 GPU 执行路径。

2026-09-11 Python 容器闭包继续收窄：私有 lowering 以无分配能力的 `BufferType`/marker 描述 Infernux 外部 buffer，wheel staging 已实际删除 `lang/field.py`、`lang/_ndarray.py`、`types/ndarray_type.py`、`lang/_texture.py` 与 `types/texture_type.py`；AOT、AD、profiler、UI、SNode 作者模块继续不进入载荷，进程退出私有 lowering 后不存在顶层 `taichi` 模块。Texture 暂不属于公开 `inx.compute` 参数，未来由 Infernux 自有 Texture/RHI 描述接入，不能提前保留 Taichi 容器成为事实标准。新的导入闭包审计明确验证上述旧模块未加载，7 项真实 Vulkan Buffer/Kernel/Mesh/Physics/资源退休回归全部通过；真实 041Lab 雪场景为 240 帧平均 0.813 ms、P95 1.18 ms、控制台零错误。原生 Program/gfx/SNode tree 对象代码仍待拆除，所以不据此勾选完整 B05。

2026-09-11 原生执行层继续退役：私有前端已经删除 kernel 直接 launch、NumPy host 参数封送、返回值同步和 shell print 路径；`Program` 改由无设备的 `InfernuxCompilerProgramImpl` 提供 SPIR-V 编译与参数布局，子构建不再生成或链接 `gfx_runtime`、`gfx_program_impl`、kernel launcher 或 SNode tree manager。Infernux RHI 的 7 项真实 Vulkan 回归全部通过；可见 041Lab 场景在新编译模块上连续 240 帧平均 0.780 ms、P95 1.160 ms，控制台零 warning/error。首次冷编译仍会阻塞编辑器数十秒，必须由 B05.3 的正式准备/预热阶段解决，不能以运行期等待或旧 runtime 缓存兜底。

2026-09-11 原生载荷第二轮深裁：`Program` 不再拥有 device、allocation、ndarray、ArgPack、Texture、SNode tree、同步、materialize 或 kernel launch 生命周期；原生绑定删除对应容器、SNode/Axis/Mesh 作者面和崩溃/线程等测试入口，构建目标同时排除 argpack、ndarray、texture、profiler、SNode host accessor 与相关 pybind 源。`ProgramImpl` 也已从完整后端虚接口收缩为编译器、目标能力与参数布局三个职责，不再声明失败即报错的 runtime 空实现。未启用但仍会维护 `ticache` 文件、锁、SHA 文件键与清理策略的上游 `KernelCompilationManager` 已从活动构建和调用链删除；随后一并删除 `CompileConfig`、`Function`、`Kernel` 上仅服务于上游 offline cache 的字段、序列化和 key，以及 offline-cache 分析/文件管理源码。引擎 Python 层现有有界 specialization/executable cache 是唯一内存复用主干，后续磁盘缓存也必须由引擎生命周期统一拥有。正式 `infernux_gpu_jit_compiler` 主 CMake 目标直接重建并安装到 wheel staging，私有原生模块由 10,637,824 bytes 降至 9,638,912 bytes（约 9.4%）；不是手工复制文件。8 项真实 Windows Vulkan Buffer/Gizmo/Kernel/Mesh/Physics/资源退休回归全部通过，CPU compute/JIT 定向回归 64 项通过。可见 041Lab 雪场景使用最终正式载荷进入 Play 约 175.7 ms，控制台 0 warning / 0 error；上一正式切片的 240 帧完整帧平均 1.522 ms、P95 2.791 ms，Game-only 平均 0.691 ms，同轮仍远高于 300 FPS 门槛，但这里只作为编译器裁剪无回归证据。IR/变换依赖、Python 闭包和引擎正式缓存/预热/跨平台发行尚未完成，因此本轮不提前勾选 B05 总项。

2026-09-11 引擎准备与项目缓存切片：`inx.compute.prepare(kernel, params)` 在组件 Start/资源准备边界只编译 specialization 并创建 Infernux RHI pipeline，不上传参数、不提交 dispatch、不试跑玩法。自动网格法线/切线 kernel 也在其拓扑资源建立时准备。编译器导出稳定参数布局后，参数块由 Infernux 直接封装；Taichi `LaunchContextBuilder`、Kernel launch context、返回值读取和对应原生源码/pybind 已删除，私有模块进一步降至 9,538,048 bytes。SPIR-V、task/binding metadata、参数布局和 domain 参数现可脱离 Program/Kernel 对象序列化为 `.inxgpu`；唯一磁盘制品由引擎写入 Editor 项目的 `Library/Artifacts/Compute/`，与 Mesh、Texture、Particle 等项目派生制品统一管理，为后续 Cook/打包建立单一来源。Player 运行期新生成的缓存才写入该游戏自身的 `persistent_data_path/Cache/Compute/`。两者均限制为 128 项/256 MiB，key 只覆盖 kernel AST、buffer schema 与 Vulkan 编译目标，不 hash 运行数据，也不使用 Taichi `ticache` 或用户主目录。真实 041Lab 清空旧位置后首次 Play 为 757.7 ms，完整关闭编辑器后再次 Play 为 183.9 ms，`Library/Artifacts/Compute` 保持 3 个文件/29,674 bytes，证明跨进程命中而非重新生成；控制台 0 warning / 0 error。缓存与准备生命周期已闭合，但 Infernux 自有 HIR、剩余 Python/IR 闭包和最终发行矩阵仍未完成。

- [ ] 引擎 kernel 前端提供已定义的单 work-item 代码/类型化 IR、执行域、buffer 布局与目标能力；裁剪模块只负责必要的合法性诊断、IR 优化、Vulkan compute 可执行的 SPIR-V 生成。这里“翻译成 vk”指着色器代码生成，不是由编译器调用 Vulkan 执行。
- [ ] 输出 SPIR-V、入口/工作组信息、参数与资源绑定布局、所需设备能力和源位置诊断映射。复用引擎权威类型/布局，不再对同一产物二次反射、猜参数或建立影子 schema。
- [ ] 当前仅 JIT：输入代码/受管 IR，运行期编译并由引擎执行。删除 AOT 导出、特化收集、module loader、独立 C-API 和 Player AOT 绑定要求；不能依赖 Taichi Program/SNode Runtime 运行代码。JIT 编译缓存不等于 AOT 交付模式。
- [x] 编译器输入不携带活动 VkDevice/queue/buffer、可修改场景或执行回调；目标信息来自引擎已有 capability 描述。编译可发生于无图形构建进程，不初始化 GPU，不为类型特化试跑真实玩法。
- [ ] 对辅助函数/优化所需内部 IR 类型可以保留必要结构，但不得借此继续维护作者 field、设备数组、内存分配或执行状态；审查最终依赖图和调用链，而不是只给旧 Runtime 换名字。

### B05.2 必须移除的职责与源码

- [ ] 移除 taichi.field、FieldsBuilder、SNode 稀疏/树状存储管理，以及作为用户数据容器/资源所有者的 Taichi ndarray；inx.buffer 是唯一引擎数据所有权入口，不维护两套主机/设备镜像。
- [x] 私有 Python 前端不再安装 `field.py`、`ndarray.py`、`snode.py`、`aot/`、`ad/`、`profiler/`、`ui/` 等作者/运行时模块；以 Infernux 自有 kernel frontend 名称承载所需 AST lowering，用户进程不注册或暴露顶层 `taichi` 包。删文件前先从编译导入闭包解耦，不能靠安装排除制造运行期缺模块。
- [ ] 原生 `taichi_python` 改为 Infernux 私有编译模块和最小导出表；删除 image I/O、SNode/Field/Ndarray/AOT Graph、Texture、稀疏矩阵/求解器、profiler/timeline、测试接口及与编译输出无关的 pybind 导出。最终原生模块不得继续链接 gfx runtime、AOT builder/loader、SNode tree manager 或独立 Program 执行实现。
- [x] JIT 改为无设备编译服务：按 Infernux kernel source/HIR、buffer schema、work-item 域和 Vulkan capability key 有界特化，返回 SPIR-V 与绑定/诊断元数据；缓存不 hash 运行数据、不做联网安装/自修复，不使用 Taichi offline cache 目录散落用户磁盘。（2026-09-11：编译产物脱离 `Program`/`Kernel` 对象，以 `.inxgpu` 存放于项目 `Library/Artifacts/Compute`；Player 新编译结果只进入自身持久化目录。真实跨进程命中和禁用私有编译器后的 Vulkan pipeline 重建均通过。）
- [ ] 移除 ti.init、独立 Program 执行生命周期、kernel launch runtime、内存池、设备创建、队列、命令提交、同步/回读及资源释放路径。必要的编译上下文改成无设备的编译服务，不保留有副作用的旧初始化。
- [ ] 移除顶层 for 自动 offload 语义，使用 B03 的单 work-item/显式执行域 lowering；复用需要的 IR/codegen，不把普通循环偷偷并行化。
- [ ] 继续裁 Taichi CPU/LLVM、CUDA、Metal、OpenGL、DX、GUI/GGUI、自动探测/回退/安装自修复及与本目标无关的可微/稀疏求解等入口；共用优化先解耦和替代，再删除，避免误删 SPIR-V 编译依赖。CPU JIT 的 Numba/llvmlite/LLVM 不在此裁剪范围。
- [ ] 沿 field/SNode/ndarray/Program/RHI/打包依赖建立删除映射，最终检查源码构建目标、链接库、公开导出及 wheel 文件清单。只隐藏 Python 名称、关闭开关而仍发布完整旧运行时不算完成。
- [x] 先贯通引擎 Buffer/RHI 执行路径再退役旧适配，迁移完成删除旧 provider/resident/runtime 分支，不留长期双栈或失败后回旧 Taichi 的兜底。

### B05.3 wheel、构建与生命周期迁移

- [x] 完成 external/plugins/infernux_taichi → external/taichi_for_infernux 的全部引用迁移。源码、子模块登记、CMake、测试、wheel staging、远程仓库名/描述/topics/README 均使用新位置；非历史计划范围已无旧插件目录引用。
- [x] CMake 直接将编译模块及必要依赖产出到引擎 wheel 的正式包载荷目录；同步 Windows/manylinux 构建、wheel 修复/依赖打包、安装测试及版本/ABI。不再以插件 Package 或 infernux_package 目标作为最终交付。（2026-09-14：Windows Release `package_python` 直接安装私有 GPU JIT 到 wheel staging；干净目标目录安装 smoke 通过，wheel 内含 `taichi_python*.pyd` 与 `licenses/taichi/NOTICE`。）
- [x] 删除该插件的 manifest/preload/provider 发现链、官方 catalog 条目、自动推荐、安装按钮及项目依赖要求；引擎 compute 接唯一内置编译入口，不保留“未装插件请安装 Taichi”的错误。
- [ ] 旧项目内 Taichi 插件在一次性加载/升级边界识别并提示迁移，避免其旧生命周期重复注册；保留用户自定义内容，不递归删除整个 Packages 目录，不通过另一 fallback 启动旧 provider。
- [x] 引擎现有启动/准备阶段拥有编译服务；wheel 随附并不意味着 import Infernux、创建 buffer 或每帧绘制都要初始化编译器。首次关键交互前显式准备/预热，不使用插件 InxPreload，也不另造平行引导系统。
- [x] wheel 安装损坏、目标能力不支持、编译错误分别报告，不能伪装为未装插件，不自动 pip 安装/自修复。引擎原生模块升级遵守现有进程生命周期，不强行热卸载；普通项目插件的 Python 刷新验收仍保留。（2026-09-11：新增 `ComputeCompilerError` 承接私有编译器载荷缺失/初始化失败；设备能力、作者 `KernelCompilationError` 与提交 `ComputeExecutionError` 保持独立且无重试。正式 wheel 干净环境 Vulkan smoke 通过，普通插件 Python 发布/预载/候选事务回归 63/63 通过。）
- [ ] 适用的 Player 统一携带 CPU JIT（Numba/llvmlite 与必要依赖）和 GPU JIT 编译模块；删除默认 AOT/可选 GPU JIT 的双模式分发。实际平台能力必须验证，不能用主机编译器二进制充当其它架构的运行载荷。
- [x] 以干净环境安装最终 wheel 验证无需项目插件/全局 Taichi 安装即可编译；删除旧插件路径后编译和运行均成立。无 GPU 的构建进程验证可以生成目标编译产物，实际执行由目标引擎验证。（2026-09-11：正式 `package_python` 产出的 Windows wheel 通过内置校验；全新 venv 中未安装顶层 Taichi、未设置源码 `PYTHONPATH`，从 `site-packages` 成功编译并执行真实 Vulkan kernel。唯一 `.inxgpu` 写入隔离项目的 `Library/Artifacts/Compute`。）
- [ ] 测量 wheel 增量、初始化延迟、编译延迟、内存、最终双 JIT Player 载荷；性能不达标优化单一内置主路径，不恢复插件或 AOT 方案掩盖结果。

## 7. B06：CPU/GPU 双 JIT 构建、部署与 042 交接

2026-09-11 旧执行双栈退役：公开 `compute.hpc`、`resident/upload/download/dispatch`、NumPy `batch`、旧 GPU provider 注册链及尚未实现的 `compute.nn` 占位入口已从实现、stub 和测试中删除；`jit_runtime` 不再保存 resident scope。源码静态识别和 Player cook 只把 `@inx.jit.compile` 视为 CPU JIT 声明，`@inx.compute.kernel` 不进入 CPU auto-parallel rewrite；关闭 JIT 载荷时直接拒绝 CPU JIT 声明，不退回 Python 执行。Lab 当前脚本已经只使用 `inx.buffer + compute.kernel/launch`。定向回归 383 passed、1 skipped。该证据完成旧作者路径退役，但尚不证明最终 wheel/Player 与各平台双 JIT 载荷，故 B06 总项保持未完成。

- [ ] 将原来以 Numba/llvmlite 为主的 JIT 依赖收集改为引擎统一双 JIT 载荷：CPU 仍含 Numba/llvmlite、必要 NumPy 等依赖；GPU 含内置编译模块、必要前端/IR/codegen 依赖与许可。不是只复制两个 DLL，也不把本机整个 conda 环境带入。
- [ ] 核对 pyproject.toml、engine/_build_dependencies.py、prebuilt_runtime.py、nuitka_builder.py、game_builder.py、build/contracts.py、compute_preparation.py、平台 Runtime 装配与 CMake install/wheel staging。原 enable_jit/numba 开关和 UI 改为一致含义，不能 GPU 沿旧插件偷渡或关闭 JIT 后仍执行编译。
- [ ] CMake/CI 直接生成并安装双 JIT 所需载荷；默认不构建 Taichi C-API/AOT targets、不生成插件 .inxpkg。按依赖图删除 AOT 专属源/绑定/metadata，而非只在 CMake 关闭仍把全套代码发布。
  - 2026-09-09 Windows 局部完成：主仓独立子构建直接产出编译模块并安装至 `PythonWheel`；wheel 审计强制检查单一私有原生模块、LICENSE/NOTICE 与 NumPy/Numba/llvmlite 元数据。当前真实 wheel 为 25,333,545 bytes，其中 `Infernux/_compiler` 为 67 个条目、压缩后约 4.28 MB；没有顶层 `taichi` 包。为贯通公开 JIT，wheel 暂带私有 lowering 源码；已删除 examples/UI/AOT/AD/graph/linalg/sparse/tools 等明显无关分发目录，但 `lang` 导入闭包仍包含 field/SNode/ndarray/profiler，原生链接也仍含 Program/gfx/AOT 相关对象，不能把深裁或本项勾成完成。
- [ ] 移除 AOT prepare_build、从已加载编辑器模块收集特化、AOT cook/loader/profile 与 build UI 依赖。构建不以“编辑器已经跑过某个 kernel”为成功前提。
- [ ] Editor/Player 从同一受管作者代码或类型化 IR 编译：源码位置/GUID、辅助函数和依赖代际随 Content.inxpkg 保存，编译器通过引擎入口读取，不依赖外部明文 Assets、inspect.getsource 的本机路径或作者工程存在。CPU JIT 需要的字节码/类型信息亦保持可用。
- [ ] 迁移 hpc/resident/Lab/stub/文档和旧 AOT 测试到 JIT；原测试只保留必要历史基线，不留一条隐藏 AOT fallback。JIT 缓存按版本/签名/目标受管并有界，缺失正常重新编译，不在执行错误后更换后端。
- [ ] 支持平台的最终 wheel 与 Player 离线验证 CPU、GPU 首次编译、重复调用、helper 更新与缓存失效；无源码 checkout、无项目 Taichi 插件、无系统 SDK/CMake、无临时下载。跟踪依赖和动态库全部来自正式分发位置。
- [ ] Windows/manylinux 使用最终制品实测；Android 必须验证 ARM64 JIT 编译器、Python/Numba 能力及驱动，不再用 Vulkan AOT 成功代替；Web 原预编译交付假设取消，需重新确定可用 JIT 路线与成本。若这些平台目前不能支持双 JIT，明确列为技术阻塞/待用户确认范围，不能恢复 AOT 或 CPU 慢循环冒充完成。
- [ ] 同测干净安装大小、wheel 下载量、Player 总量、CPU/GPU 各依赖大小、首次编译/预热和稳定运行。JIT-only 去掉 AOT 开发链可能减小实现规模，但 Player 携带编译器可能更大；不预先宣称缩小。
- [ ] 042 复用两类 JIT 和 buffer 代际/退休。普通动态 Python 与编译新 CPU/GPU kernel 能力分别控制；同一编译服务执行 authored 与动态代码，不另建发布系统。线程及代码资源在途时不能提前释放。
- [ ] Apache LICENSE/NOTICE 与原作者标记按实际引擎 wheel/平台 Runtime/Player 分发保留；CPU JIT 的 LLVM/Numba 许可同样收集。名称更改不改变上游归属。

## 8. 顺序与验收

1. 冻结 B01/B03 公共语义、B05 编译器输入输出/内置迁移清单与 B06 双 JIT 发行合同，记录旧路径基线；只做计划不会自动恢复暂停的实现。
2. B01 原生 buffer/传输/生命周期 + B03 单 work-item 最小闭环，验证原地计算与无隐式回读。
3. B02 动态 Mesh/GPU 法线/渲染直连，继而物理批量交换；引擎 GPU 化审计覆盖其余消费者。
4. B04 CPU buffer 接入、新装饰器、数据流分析、优化分层/自动串并行/失效及 V8 对照。
5. B05 纯编译模块裁剪与 wheel 内置迁移、B06 构建/平台/迁移/署名，交付单一成熟主路径。

- [ ] 保留 A03.4 原验收：内置 Sphere 变形、真实文件材质、普通 Rigidbody 双向交互、真实编辑器 Play 持续交互 300 FPS+；1/8/32/128 刚体的近场/远场分别测量。不能靠静止、减少子步、关 Gizmo 或隐藏 CPU 往返宣布达标。
  - 这里的双向交互是硬门槛：刚体必须被软体反作用弹开/转动，软体也必须因刚体接触产生形变与速度响应；使用引擎自带 Rigidbody/Jolt 世界，不允许脚本改 Transform 冒充。刚体数量增加时远处无关物体不得扩大软体全遍历，整体完整帧目标约 300 FPS。
- [ ] 不承诺任意数量密集接触时耗时恒定；记录随规模增长的 solver/宽相/传输/绘制成本，优化不合理的全量扫描。性能目标失败仍标未完成，不改变验收对象。
- [ ] 类型化 buffer 与等价手写 shader 在同任务/同精度/同设备下对照；首次准备与稳态计时分离，评估编译后代码质量和调用开销。
- [ ] 新接口所有勾选以测试/最终消费者证据为准；旧 resident/hpc 原型的既有通过项只作为迁移回归基线，不折算为新接口完成比例。

## 9. 参考

- [V8 Maglev：分层编译、SSA、类型反馈与去优化](https://v8.dev/blog/maglev)。参考其机制，不照搬整套 JavaScript VM。
- [Taichi ndarray 与 field 的数据布局边界](https://docs.taichi-lang.org/docs/master/ndarray)。借鉴类型化访问，不强制使用上游容器。
- [Unity Mesh.GetVertexBuffer](https://docs.unity3d.com/ja/current/ScriptReference/Mesh.GetVertexBuffer.html)。计算与渲染共享 buffer 仍需明确布局和资源用途。
