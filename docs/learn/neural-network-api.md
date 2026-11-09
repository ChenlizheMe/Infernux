# 神经网络 API 支持范围

神经网络统一从 `infernux.nn` 使用。CPU 与 Vulkan 共用静态模型定义、参数状态和编译工作描述，运行时不依赖 Torch。函数名与已支持参数采用 Torch 的写法；下面明确列出当前可执行的范围及限制。

## 模型与张量

| API | 当前行为 | 支持范围 |
| --- | --- | --- |
| `nn.tensor(data, dtype=None, device=None, requires_grad=False)` | 创建拥有自己数据的张量 | `float32/int32/uint32/int64`，设备为 `cpu` 或 `gpu`；默认 CPU、float32，复制 Tensor 时保留原类型；不支持空张量 |
| `nn.Tensor`、`nn.Parameter` | 数值张量、注册参数 | 张量复用引擎 Buffer；参数与梯度使用 float32，整数值不参与求导；Parameter 随模型统一发布，不支持跨独立模型共享参数所有权 |
| `Tensor.shape`、`dtype`、`device`、`numel()` | 查询逻辑形状、类型和设备 | 不触发数据读回 |
| `Tensor.numpy()` | 取得独立 CPU 值 | 直接用于 CPU；GPU 必须先明确执行 `detach().cpu()`，该传输同步读回 |
| `Tensor.item()` | 取得单元素的 Python float 或 int | 保留整数精度；GPU 标量会在此调用中同步读回 |
| `Tensor.detach()` | 创建不参与求导的同值视图 | 不复制数值载荷 |
| `Tensor.to(device=None, dtype=...)`、`cpu()`、`float()`、`int()`、`long()` | 取得目标设备或类型上的值 | 同设备同类型返回自身；传输保留原类型，类型转换运行共同原语；转为整数停止梯度；当前不支持跨设备反向边 |
| `Tensor.as_buffer()` | 取得同一存储的只读平坦视图 | 不读回 GPU，不复制显存内容 |
| `nn.Module` | 通过 `forward(self, observation, actions, ...)` 定义静态模型 | 一个或多个张量输入，返回单个张量或一组有序张量 `tuple`。输入的形状、类型、设备和梯度签名共同确定固定图，同一模型的图共享参数与优化器状态 |
| `model(*inputs)`、`model.forward(*inputs)` | 执行同一已捕获模型 | 各浮点输入按 requires_grad 接收梯度；重复传入同一叶张量会合并贡献，未使用输入不产生梯度 |
| `nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)` | 仿射层 | float32 二维输入 `[batch, in_features]`；带偏置或不带偏置；整数特征先显式 `.float()` |
| `nn.Conv2d(in_channels, out_channels, kernel_size, ...)` | 固定 NCHW 卷积与输入/权重/偏置反向 | 正整数 stride/dilation/groups、可选 bias、零填充；padding 支持整数/二元组、valid、单位 stride 下的 same（含非对称边界） |
| `nn.PixelShuffle(upscale_factor)`、`nn.functional.pixel_shuffle` | 通道重排放大与反向 | 固定 NCHW，通道数须整除放大倍率的平方；无重采样近似 |
| `nn.Sigmoid()` | 稳定 sigmoid 与反向 | 正负输入分别处理指数；CPU 原语与 GPU 局部 AD 共用图上的梯度约定 |
| `nn.ReLU(inplace=False)`、`nn.SiLU(inplace=False)`、`nn.Tanh()` | 激活层与反向 | CPU/Vulkan 共用算子定义；GPU 使用原生局部 AD，当前不支持原地写入 |
| `nn.Softmax(dim=None)`、`nn.LogSoftmax(dim=None)` | 指定轴归一化、对数概率与反向 | 减去轴最大值避免指数溢出；支持正负轴，跨元素梯度通过同一图上的显式 VJP 执行 |
| `nn.functional.relu/silu/tanh/sigmoid/softmax/log_softmax` | 对应函数形式 | 可用于模型捕获和独立叶张量；当前不把两份已编译模型的结果自动拼接为新反向图 |
| `nn.Sequential(*modules)` | 顺序组合 | 也支持 OrderedDict 定名和按下标读取 |
| 编译区域内的 `+`、`-`、`*`、`/`、负号 | 逐元素算术、标量常数、广播 | 支持共享值与共享层的梯度合并；标量除数非零；不是任意 eager Tensor 运算 |
| `nn.cat(tensors, dim=0)`、`nn.stack(tensors, dim=0)` | 捕获图中的拼接、堆叠与反向拆分 | 多个图值，正负轴；当前在 Module 捕获区内使用，不支持 out 写入 |
| `nn.reshape(input, shape)`、`Tensor.reshape(*shape)` | 元素数不变的形状变换 | 支持一个 -1 推断维；目前通过共同复制原语物化，不承诺别名视图 |
| `nn.flatten`、`Tensor.flatten`、`nn.Flatten` | 合并连续轴 | 正负起止轴；Module 默认从第 1 轴开始 |
| `nn.transpose`、`Tensor.transpose`、`nn.permute`、`Tensor.permute` | 交换或排列轴及反向逆排列 | 支持任意受控阶数与正负轴；首版物化连续输出，不承诺 stride 别名视图 |
| `nn.sum/mean`、`Tensor.sum/mean` | 单轴、多轴或全部轴归约及反向扩展 | 支持 keepdim 和 float32；None 或空维度列表均归约全部轴 |
| `nn.argmax`、`Tensor.argmax(dim=None, keepdim=False)` | 选择最大值的首个位置，输出 int64 | 支持整体或单轴、负轴、keepdim；并列取首项、NaN 取首个 NaN；不求导，扫描目前在完整输出行之间续行 |
| `nn.sin/cos/square`、对应 Tensor 方法 | 常用数学运算与反向 | 在捕获图内组合，或独立叶张量调用；CPU/Vulkan使用同一原语定义与专用VJP |
| `Tensor[...]` | 固定基本切片和梯度回填 | 整数、负整数、正步长 slice、None、ellipsis；不支持空结果、负步长或张量索引；结果物化连续存储 |
| `nn.functional.interpolate(input, size=..., mode=...)` | NCHW nearest/bilinear 采样与反向 | 支持显式正整数 size、整数 scale_factor，或 recompute_scale_factor=True 的非整数比例；bilinear 支持 align_corners；暂不支持 antialias 或 bicubic |
| `nn.functional.grid_sample(input, grid, mode=..., padding_mode=..., align_corners=...)` | 图像与位移场组合采样及两个输入的 VJP | 在 Module forward 中组合 NCHW 图像与 NHW2 网格；nearest/bilinear、zeros/border、两种 align_corners；暂不支持 reflection、bicubic、体采样或两个独立 eager Tensor 的自动反向图拼接 |
| `nn.squeeze/unsqueeze`、对应 Tensor 方法 | 移除单位轴、插入单位轴 | squeeze 支持单轴或多轴；通过共同 reshape 物化 |
| `nn.ones_like/zeros_like` | 创建同形、默认同类型的值 | 编译区域生成原生常量填充节点；普通调用创建独立叶存储；支持显式 dtype，默认不继承输入梯度 |

捕获允许由 Python 常量决定结构；张量数值不能决定 Python 分支或循环。离线调用遇到新输入签名会捕获对应图，重复签名直接复用；帧内受管请求只使用事先准备好的图，不在调度时编译。模型结构改变后调用 `recompile()` 清空图缓存；参数拓扑改变时应创建新模型。模式切换和设备迁移也会清理相关图，不创建参数副本来模拟多种批量大小。

图像重采样按轴分解，共享普通值图和 VJP；反向的归约仍由编译器生成合法内切点，可以随训练预算跨帧续行。没有为图像另建一个执行器。工作项中的 `float(...)` 表示显式 float32 转换，CPU 与 Vulkan 使用相同数值契约。

grid_sample 前向使用真实的邻点 gather。动态浮点地址只在显式处理 NaN、夹到可精确表示的静态上界后转为整数，仍检查外层整数地址范围。zeros 复用一圈补零，border 使用边界夹取；nearest 的平局按原图像坐标取偶数。反向首版对图像使用确定性的逆向 gather、对网格沿通道归约，均可沿已有切点续行，不要求浮点原子。图像反向成本约为输入像素数乘输出像素数，适合当前受控小网络验证，大图训练仍需降低复杂度；不能把这条正确性闭环视为大图性能已经达标。

归约的空维度列表含义沿用 [PyTorch 原生归约规则](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ReduceOpsUtils.h)，与 NumPy 的空 axis 元组不同。

绑定的子层可以直接调用或 `prepare()`，仍使用所属根模型的参数域、锁和梯度槽。切换根图、子层图和批量大小不取消其他仍然有效的图；真实参数更新会使旧参数代的反向结果失效。完整参数加载、优化器和模型文件仍以根模型为边界。导出文件封装根模型最近选定的一种签名；需要其他签名时先 `prepare()` 再导出。

## 状态与训练

`with nn.no_grad():` 和 `@nn.no_grad()` 暂停梯度记录。外部推理不保留反向激活；它不会把同签名的后续训练永久变成无梯度调用。模型 forward 内部的 no_grad 和 `value.detach()` 捕获为真正停止反向依赖的节点，保存到 INM 后依然成立。未使用参数保持 None 梯度，Adam 不为它们推进动量和计数。无梯度输出仍可计算 MSE/稠密交叉熵损失值；同设备模型输出复用其已准备或烘焙的目标程序，跨设备复制后不携带原设备的程序来源。

| API | 当前行为 | 支持范围 |
| --- | --- | --- |
| `parameters()`、`named_parameters()` | 按注册顺序枚举参数 | 递归、去重枚举；不支持关闭递归或关闭去重 |
| `state_dict()` | 取得命名参数快照 | 保留旧参数存储；之后训练发布新参数不会改变旧快照 |
| `model.parameter_version` | 查询本实例的参数发布计数 | 子层共用根计数；完整加载与实际优化器发布递增，准备、清梯度、取消未发布更新和同值设备迁移不递增 |
| `load_state_dict(state, strict=True, assign=False)` | 完整验证后发布参数集合 | 全部名称、形状和类型匹配；不支持部分加载或替换参数对象 |
| `train(mode=True)`、`eval()` | 设置捕获模式 | 模式改变会重新捕获；尚无 Dropout、BatchNorm 等模式相关层 |
| `Module.to(device)`、`cpu()` | 移动完整参数域 | 设备名为 `cpu`、`gpu`；不使用 Torch 的 `cuda` 设备名 |
| `nn.functional.mse_loss(input, target, reduction="mean")` | 均方误差标量 | 可求导输入为模型输出，目标不可求导；无梯度评估也接受普通 Tensor；形状与设备相同 |
| `nn.functional.cross_entropy(input, target, ignore_index=-100, reduction="mean")` | 稳定交叉熵与反向 | 支持 float32 概率目标或 int64 类别目标，mean/sum；类别轴为第 1 轴，单向量为第 0 轴；整数目标支持完整 int64 ignore_index；暂不支持 weight、reduction="none" 或平滑 |
| `loss.backward()`、`output.backward(gradient)` | 反向传播并累积参数梯度 | MSE/交叉熵使用默认种子；普通输出可传同形、同设备的 detached 梯度，单元素可省略；不支持高阶梯度和保留反向图 |
| `nn.optim.SGD(model.parameters(), lr=...)` | 普通 SGD | 一个完整模型参数域、一个存活优化器；当前不支持动量、权重衰减、参数子组或 closure |
| `nn.optim.Adam(model.parameters(), lr=..., betas=(.9,.999), eps=1e-8, weight_decay=0)` | Adam 与耦合 L2 衰减 | 参数、两组动量和逐参数计数一起提交；None 梯度跳过，零梯度正常更新；暂不支持 AMSGrad、foreach、fused 或 AdamW |
| `optimizer.zero_grad(set_to_none=True)` | 清除或置零梯度 | `True` 释放梯度，`False` 执行已编译置零操作 |
| `nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None)` | 按全部参与梯度的 L2 范数统一缩放，返回裁前范数 | float32、单设备、同一模型域；当前只支持 L2、非有限值传播和 foreach=None/False。忽略 None 梯度；零梯度保留 |
| `optimizer.step()` | 完整更新后一次发布参数 | 离线调用同步完成；执行中间结果不会作为半套新参数暴露 |

Categorical 已交付；桌面 NNNP1 项目中的一步 actor-critic 使用普通图组合损失，没有专用 RL 执行器。它已验证单动作/多关节动作的 TD0、停止梯度、参数发布与终止/截断数学；Microduck 物理学习场景及连续策略另行验收。卷积当前是可续行的直接求和实现；尚未做专门的分块、融合或卷积库加速。不能把首版图像算子称为完整超分或插帧算法。

公开 Tensor、静态 Plan、CPU 原生入口、Vulkan、INM 和预编译产物均保留 float32/int32/uint32/int64 真实类型。整数支持复制、切片、形状变换、拼接和显式类型转换；浮点数学、参数和梯度仍使用 float32，整数不进入梯度域。`softmax/log_softmax/sum/mean(dtype=nn.float32)` 会先转换整数输入，不支持的 dtype 明确报错。多个输入可有不同类型，离线模型调用时位于同一设备，受管 Program 可分别传入 CPU/GPU 来源。`argmax` 和 Categorical 采样输出真正的 int64。

受管请求的 `parameter_version` 记录接单时的参数版本。`program.submit(..., expected_parameter_version=v)` 在同一模型锁内检查版本，再进行来源采样、IO 和请求创建，可用于拒绝旧策略轨迹。推理持有接单时完整、不可变的参数存储；即使训练或完整加载发布了新参数，尚未开始或已部分执行的推理都继续使用原来那一代，不混用参数，也不因普通换代被取消。新请求使用新发布的一代；结构重新编译或设备迁移仍会使旧执行配方失效。训练写入则继续检查参数、梯度和优化器的当前身份，避免旧写入覆盖新状态。

训练请求上的版本保持为输入策略版本，成功发布后模型版本递增。版本只在同一模型实例内比较，新加载的模型从 0 开始；它不是文件格式版本，也不是优化器步数。应用还需保持动作策略与环境语义一致：本例的 on-policy 训练器拒绝 eval 贪心模式，批次版本应来自成功推理请求，最终观察应在 reset 之前封存。

SGD 和 Adam 使用同一更新操作构建、续行和发布路径。受管训练在请求释放时固定本次超参数与旧优化器状态；后续修改参数组只影响下一次更新。检查点包含 Adam 的两组动量和每参数计数，跨设备恢复和预编译模型训练也使用这份状态。取消未发布的更新不会把候选动量或半套新参数暴露给调用者。

损失程序按目标及值/梯度角色统一准备；交叉熵的 log-softmax 临时数据、归约、反向和更新进入同一受管游标、容量和成本描述。稠密目标的公式保留其沿类别轴的总和，不暗中假定输入已经归一。目标的概率含义由作者提供，当前不在 GPU 主路径中读回并重新解释目标。

## 编译、帧内执行与模型文件

`model.prepare(*example_inputs)` 是显式冷准备入口。CPU 会准备原生执行入口；GPU 会准备受支持的前向程序。普通 `model(x)`、`loss.backward()` 和 `optimizer.step()` 是同步离线接口，调用本身不提供帧预算保证。

`@nn.program` 支持叶/已 detach 张量和声明的场景字段输入，记录单模型推理，或 `zero_grad → model → loss.backward → optimizer.step` 训练正文。损失可用 MSE、概率/整数类别交叉熵，也可以直接使用模型 forward 返回的零维标量；后者用单位梯度反向，不需要伪造目标或再平方损失。准备时只记录结构，不执行推理、字段采样或更新。训练可以返回 `loss.detach()` 或 `None`；推理返回预测或其 detach。当前不支持在区域中组合多个独立模型或任意 Python 数值运算，预处理和自定义损失放进模型 forward。

`program.prepare(*inputs)` 冷准备对应签名；普通 `program(*inputs)` 同步运行已记录配方，不逐次执行作者正文。`program.submit(*inputs, finish_within_frames=..., budget=..., result_device=...)` 把已准备配方交给当前引擎，返回带 `done/status/reason/result/metrics` 的现有请求；未准备的签名在提交时报错，不在帧内编译。CPU来源到GPU模型和反向读回沿共同传输管道。`nn.Budget` 直接公开现有共同预算配置，包含宿主额度、CPU/GPU/IO时间目标、容量、分块和并行上限，不给每个program额外开一份全局额度。

`finish_within_frames` 默认是 `None`：请求按每次机会的预算续行，保留同一个图、游标和私有中间状态，直到完成、主动取消或来源失效。适用于持续训练等没有结果硬截止的工作；不为它建立有限期完成预约。指定正整数才建立硬截止，超期仍沿原取消和真实退休流程处理，未完成更新不会发表。普通 `compute.Continuation.submit` 使用同样的规则。有截止与无截止工作共存时，已预约的有限批次先完成，随后无截止队列获得执行机会，新的有限请求可以继续建立预约。

`cpu_slice_ns/gpu_slice_ns/io_slice_ns` 影响片段大小，`cpu_frame_ns/gpu_frame_ns/io_frame_ns` 分别约束本宿主帧已受理的预测执行量。`transfer_bytes` 默认每段最多 1 MiB，`transfer_frame_bytes` 默认每个宿主帧最多受理 8 MiB 的 GPU 传输。可分段的传输根据剩余字节额度、GPU 实测耗时和可用工作空间选择实际大小，暂存页也计入工作空间。主线程准备使用 `owner_ns`，驻留存储使用 `capacity_bytes`，文件读写使用独立的 IO 额度。所有请求共享同一个原生账本。

GPU 传输账包括普通 compute FIFO 与受管程序的真实上传、复制和回读，也包括每次编译程序启动所需的参数块。普通 FIFO 与受管提交各自在真正受理时记一次账；Busy 不占额度，完成后按原提交宿主和原 grant 结算。取消已经受理的回读消费者不抹掉实际发生的流量。参数块是一次启动不可拆的传输单元，首项超额只影响传输字节这一项，不放宽数值计算的 GPU 时间目标；同帧后续提交继续检查剩余额度。

图形准备固定实际使用的显存分配；图导入、材质描述符与共享 work 的绑定持有这份存储，消费依赖也从同一分配读取最新已受理的写票。共享 work 重绑定生成新的 dispatch，复用原有私有工作缓冲；旧绑定沿真实 GPU 完成关系退休。这保证地址和依赖一致，尚不提供跨帧内容冻结：普通只读 Buffer 视图仍然不是数值快照。

普通同步求值、显式同步 flush 和冷准备保持原同步含义，并记录实际接受及超额；它们不创建额外宿主帧，也不提供后台帧预算保证。可延期的普通 FIFO 则与受管任务共享当前宿主的 GPU 和传输准入。画像分别记录主线程准备/提交、GPU 实际时间和传输字节，不把携带参数的数值 dispatch 归类成纯传输。

真实受理才扣预测额度；未知操作一资源一帧只获得一次校准，不能按零成本重复提交。同帧真实完成票可以将该次预约替换为实测成本；跨帧完成只更新累计实测和后续画像，不改变新帧额度。取消本身不返还额度。已知不可拆的最小操作可在空闲资源上受理一次超额工作并记录，不改变物理步长，也不宣称操作可以被随时中断。校准探测也必须装入帧额度，传输封包前按帧额度分块并保持元素对齐。

当前账本已接受管 CPU/GPU/模型 IO 和原 compute FIFO。原生协调器的同帧多轮服务、期限日历对累计额度的完整预测，以及普通预览/资产工作源的逐阶段接入仍在重整中；上述账本不代表所有引擎 IO 已完成统一准入。

GPU单操作耗时只由该操作的真实回执更新；多操作命令包的计时只校正对应组合的打包选择，不扩大其他操作或传输的每行成本。单一尺寸无法区分启动成本与每行成本时，会独立执行有界的另一尺寸以校准。命令包和宿主准备成本使用近期样本，过期的慢样本不会永久压低分段粒度。预测在接受前冻结，回执记录原预测与实际耗时，不能事后重写当时的估计。

CPU与GPU复用同一个拓扑前缀游标。CPU可把多个已冷准备的小算子合成原JobSystem中的一张完成票，按`cpu_slice_ns`测量目标、`segment_operations`数量上限与当前宿主准备额度选择；未知耗时先独立校准。生产者完整行域结束后才接后继，局部行、内循环累加、拷贝和训练发布阶段保留切点。整段成功后才确认游标；失败或取消会废弃未发表结果，并等待真实任务退休。原生执行在算子之间检查取消，单个正在执行的入口仍不能强制中断。

CPU逐算子时间由原生入口分别测量，不将整票耗时平均摊成各层样本；排队与观察延迟仍按同一张完成票记录。有限窗口预测使用相同的CPU段选择规则合并同阶段、可完整执行的小算子；局部范围与累加段保留各自的服务单位。这里的执行时间是近期观测驱动的软目标，完成窗口也仍受真实供给与最后发表条件约束。

受管 `zero_grad(True)` 在校验成功的梯度发布点生效，清除未使用参数的旧梯度，不在提交或尚未开始时取消时修改模型状态。当前不支持 program 中的 `zero_grad(False)`。周期/事件释放接口见下文。各输入与目标按来源独立进入共同传输游标，完全相同的只读区间只传输一次并共享结果，各字段保留原有数值类型。`prepare_work(*inputs)` 把已准备的 GPU 前向变为 compute 工作描述。单输入端口为 `input`，多输入按原位置命名为 `input0`、`input1` 等，未使用的输入不产生端口。单张量输出端口为 `output`，多输出按返回顺序命名为 `output0`、`output1` 等。权重在准备时固定，权重更新后可重新准备和发布描述。

传入普通 `Material` 或 `nn.Channel.materials(materials)` 后，正文参数上的 `get_float/get_vector2/get_vector3/get_vector4/get_color` 声明一个 `[对象数,分量数]` 输入字段。正常区域的 getter 仍返回原来的 Python 数值。字段名使用字符串；多个字段可分别作为模型输入，在 `Module.forward` 内组合、转换和计算，也可独立作为训练目标。这里只降低显式传入的来源参数，不改写全局对象上的普通 Python 调用。

普通材质字段以 CPU 属性表为准。受管执行在合法线程和共同预算内，一次原生批量读取本请求的字段，写入实际被模型引用的 CPU 存储；没有第二次打包复制。数据在执行时读取，准备和提交均不读取数值。GPU 模型随后走现有分段上传，整个完成窗口包含采样、传输和计算。首版采样是有行数、字节和宿主时间上限的原子动作，超过上限应缩小批次；尚未承诺任意大场景能在一次帧机会中采完。

已有显存数据时，可以用 `nn.Channel.materials(materials, resident_fields={"metallic": nn.as_tensor(gpu_buffer)})` 声明字段的来源。GPU 张量或 `request.output_source()` 提供与材质行顺序一致的 `[对象数, 分量数]` 数据。编译区的 `mat.get_float("metallic")` 直接引用这个来源，其余字段仍按 CPU 属性表采样。结果来源尚在执行时，消费者沿原完成票等待。

`nn.Channel.tensors({字段名: tensor})` 则声明已有的只读数值字段。每个字段为 `[对象数,分量数]` 叶/已 detach 张量，且对象数一致，沿同样的 getter 入口引用；`get_int` 读取 int32/uint32/int64 单分量字段，其他 getter 读取 float32。它不分配材质采样缓冲；同设备共享输入存储，跨设备使用共同传输路径。同布局换成新 Tensor 会读取本次提交的存储，形状、类型或设备改变则重新 prepare。GPU 字段可以通过 `as_buffer()` 与渲染 storage buffer 共享同一资源；这不等于把旧材质 UBO 或任意可变缓冲自动变成神经输入。

`nn.Channel.transforms(inx.create_batch_handle(transforms))` 使用严格 Transform 批次，沿相同 getter 入口声明位置、旋转、缩放等输入。字段与普通 `inx.batch_read` 共用原生字段表，例如 `get_vector3("position")`、`get_vector3("local_scale")`、`get_vector4("rotation")`；四元数分量顺序与 Transform API 一致。程序准备只读取布局，不采样场景。受管执行按共同原子采样预算，把本次所需字段直接写入私有输入；跨帧训练和推理继续使用这次完整采样，后续 Transform 修改不会改写它。GPU 模型的上传也进入原传输管线。这是采样时的场景值，不是提交事件发生时自动保存的一份历史场景。

`nn.Channel.hinges(batch)` 同样声明 `get_vector2("motion")`，读取铰链角度和角速度，单位为弧度和弧度/秒。Transform 和关节都保留原生严格句柄校验，采样前源被销毁会使请求失败，不会换成别的对象。选择 CPU 或 GPU 不改变上述来源语义，也不会隐式启动物理。

每个引擎的原生 `FrameWorkOwner` 统一维护计算机会、宿主剩余额度和 CPU/GPU 来源轮转。数值请求、普通 compute FIFO、普通资产预览保留各自的原队列和完成票，Python 不再另起帧时钟或重复补额度。CPU 工作可以在场景更新前推进；GPU 工作要等当前前景渲染让出机会，并检查真实帧队列压力和已受理的后台工作。预览内部在导入纹理、材质、CPU图像、模型/prefab与Timeline之间轮转，每次授权只推进所选域，不让持续材质编辑占满所有机会。

宿主费用按最外层计量范围结算，嵌套的准备、IO 和物理接口只归因一次；查询余额也扣除正在运行的范围。已发生支出后，提高总额度配置在下一机会生效，不补回当前额度；显式减少前景预留可以释放当前后台份额。CPU 执行、GPU timestamp 和 IO worker 的费用仍分别观察，不把墙钟等待当设备执行时间。

这是合法动作之间的准入和续行，不会硬中断已提交的 dispatch。普通模型/prefab与Timeline的几何上传复用原资源管理器的票据，逐块提交，不再通过同步创建顶点/索引缓冲等待GPU。每个请求保留自己的不可变来源和上传游标，取消后已受理的块仍等真实退休；逻辑删除预览状态不会让GPU提前报告空闲。静态纹理查询只返回现有图片，内容变化和GUI驱逐才重新挂起，GPU缓存与异步完成分别验证原内容资格，避免把旧结果发布成新版本。

纹理读取、分块上传、普通 loader 受理和各类预览已进入共同 owner，并按真实回执结算。普通 Mesh loader 的整文件读取/反序列化、预览冷准备的完整事前预测，以及普通资产、预览、驻留和数值请求之间的共同容量准入仍未齐。这些属于第一阶段最终验收的未完成项。

## 结果的物理消费与暂停单步

`Physics.apply_result(request, targets, apply)` 在结果就绪后的物理边界应用一次。等待时沿用场景已有状态，固定步照常推进。`targets` 是目标组件序列，`apply(resolved_targets, result)` 收到当前有效的组件和结果。目标以严格句柄解析，组件或世界失效时撤销消费边。CPU 和 GPU 都能提交，CPU 需要的结果可指定 `result_device="cpu"`，回读计入原请求。

当积分必须使用这次结果时，传入 `required=True`。未就绪时固定步保留自己的执行位置，宿主仍处理 IO、完成回执和工具。多个常规结果在同一目标上按登记顺序交付，独立目标可以分别推进。必需依赖按本步要求处理。

返回的 consumer 提供 `status`、`reason` 和 `cancel()`。常规结果失败后结束交付，必需结果失败时由调用者处理并解除依赖。取消只撤销消费关系，已受理计算仍由原 owner 结算。普通协程等待只影响协程。固定步长在开始时固定，等待期间修改设置作用于下一步。`completed_fixed_step_index` 记录已完成的积分，`published_fixed_step_index` 记录已发布的物理姿态。

暂停 Step 可以跨多个宿主完成一次逻辑步骤。该步骤实际释放或明确采纳的必需工作及其真实前置依赖继承 Step 许可；普通事件不会借用这个许可，组件禁用、层级停用和显式 Scope 暂停仍有效。Step 的命令完成序列在物理结果发布、相应 gameplay 帧及必要渲染输入完成后递增，不在提交时提前报告成功。

这一入口覆盖普通组件上的物理结果应用；渲染输入使用下述渲染消费边。任意图内资源的生产者、跨系统来源冻结仍需各自的数据接口表达，不能据此宣称全部消费者自动建立依赖。

## 渲染结果的交付

`infernux.rendergraph.apply_result(request, targets, apply)` 在结果就绪后的场景抽取边界应用一次。等待时渲染沿用已有输入。回调和物理入口相同。CPU 模型可在提交时指定 `result_device="gpu"`，在回调中用 `result.as_buffer()` 交给图的 `import_buffer()`。传输计入原请求，图直接使用发布的 GPU 存储。

传入 `required=True` 时，当前抽取会等待结果。宿主继续处理输入、GUI、IO 和完成通知。就绪后先应用，再抽取并提交。收到 `render_submitted` 后登记的消费边只作用于后续抽取。图内确实必需的输入可用 `BufferInput.require(request)` 关联到具体读取方。

消费边沿用调用者 Scope、严格组件句柄和场景寿命，支持编辑工具在 Edit 状态主动计算。普通 Play 组件暂停后，其渲染应用和同目标后续应用一同挂起，画面可以继续展示现有数据；显式编辑 Scope 仍按自身状态运行。暂停 Step 中实际生命周期提交的必要渲染结果，可以沿该步骤的精确许可完成；单步保持开放直至必要结果应用和渲染阶段结束，不重放已完成的物理或脚本。等待期间的普通 GUI/事件不会仅因存在一个未完成 Step 而获得许可。

同目标的常规结果按登记顺序应用。显式必需的结果可先于仍在等待的常规结果应用。调用者若要让早先结果失效，可以取消相应消费边。失败保留原因。世界替换、组件销毁或调用者关闭会撤销相应消费边，已受理工作仍按原票退休。

神经请求统一以宿主帧作为预算和完成窗口。`nn.Clock("host")` 读取主循环服务周期；无窗口、没有物理积分或没有成功呈现图像时，宿主帧仍可继续。`nn.Clock("physics")` 记录真正完成的固定步，`nn.Clock("render")` 记录提交的前景画面；这两种钟用于触发和来源身份，不根据 CPU/GPU 选择请求期限。

`Program.submit`、`Continuation.submit` 与 `compute.transfer` 的 `finish_within_frames` 可选参数只计宿主帧。默认没有硬截止，预算不足保留同一个图与游标；指定正整数设置完成窗口。调用者 Scope 暂停期间窗口也暂停计时，恢复时顺延；它不是绝对墙钟期限。请求的 `submission_stamp` 固定记录提交宿主身份，实际观察来源仍由资源数据保存。事件、提交、完成和应用不混为一个时刻。

`finish_within_ms` 是独立的真实时间目标，适用于 `Program.submit`、`Continuation.submit`、`compute.transfer` 和模型读写。例如 `request = infer.submit(observation, finish_within_ms=8)`。它从引擎接受请求、捕获输入之前开始，用单调时间计时，Scope 暂停也不顺延。可以同时指定宿主帧窗口和毫秒目标；任一到期都会停止接受后续计算。毫秒任务及其真正前置按期限优先服务，物理必需结果仍优先；所有工作仍共享原有预算，不按 FPS 换算期限，不承诺硬实时，也不能中断已经接受的运算。

超期请求报告 `expired`，已受理的工作继续保留原票至真实退休，不能把取消当作完成或提前释放在途数据。`completion_target_ms`、`remaining_target_ms`、`wall_target_missed`、`wall_elapsed_ms` 和 `wall_lateness_ms` 记录目标及实际观察结果；后两个指标截至 owner 观察完成并退休，不代表设备内部完成时刻。训练候选到期后不发布部分参数；一次不可拆的完整发布或文件原子提交已实际完成时，保留真实结果并如实标记迟到。

`nn.export/save_checkpoint/load/load_checkpoint` 使用同名 `finish_within_frames`、`finish_within_ms` 和共同 `budget`，没有这些参数时保持同步调用。模型写入一旦实际发布便报告真实结果，`observation_late` 记录宿主窗口的通知迟到，毫秒目标另由 `wall_target_missed` 记录，不将已经发布的文件虚报为取消。请求不再接受另一套 clock/tick 期限。

准备好 Program 后，可以将释放工作绑定到真实时钟或信号。以下沿用前面的 `learn(surface, goals)`；回调返回的字典键就是原函数的参数名：

```python
physics = nn.Clock("physics")
periodic = learn.schedule(
    lambda stamp: {"surface": materials, "goals": targets},
    clock=physics, every_ticks=training_interval, overlap="skip",
    budget=shared_budget,
)

ready = nn.Signal()
event_task = learn.on(
    ready,
    lambda surfaces, goals: {"surface": surfaces, "goals": goals},
    budget=shared_budget,
)
ready.emit(materials, targets)
```

`schedule` 在实际物理、渲染或宿主相位收集周期事件。周期属于创建时的时钟世代，世界替换后需要重新绑定。每个到达的周期保存自己的 due tick；一个宿主帧有多个物理步时，各次事件仍共用当前预算。Headless 可以使用 host 周期或主动 Signal。

`overlap` 默认 `"queue"`：按事件发生顺序保留，每个绑定至多一个正在执行的请求。`"skip"` 显式丢弃忙时事件；`"latest"` 显式合并未开始处理的事件。系统不以持续布尔值代替事件计数，不通过增加事件补充预算。`pending_events/oldest_trigger/received_events/released_events/completed_events/failed_events` 用于观察积压；`skipped_triggers/coalesced_triggers/abandoned_events` 如实记录显式丢弃、合并与关闭。

收件只保留事件参数与身份，不调用作者代码。provider 在原 owner 有宿主额度和请求槽时运行，消耗同一份预算；生成的 arguments 若暂时不能提交会保留，重试不重复 provider 副作用。它仍是不可拆的作者调用，不应在其中编译或执行无界复制。提交窗口从真正提交开始，`trigger_host_frame/queued_host_frames` 单独公开排队耗时，不能将其当作包含积压的端到端保证。

保留参数引用会延长对象寿命，但不会冻结可变 ndarray、Buffer 或场景。需要历史输入时，事件应传入自己持有的样本；从场景字段采样仍发生在真实采样阶段。已有 GPU 输入可以共享，尚未受理生产的新 GPU Tensor 仍需补齐 producer 管线，不能冒充零拷贝已就绪。

`binding.completed.add_listener(callback)` 在请求真实退休后调用 `callback(sequence, request)`，逐事件取得结果与实际状态，不保留无限结果历史。`binding.request` 为最近一次请求，`last_trigger` 是触发时刻，`last_release` 是宿主提交时刻。失败、取消、过期仍有真实终态；provider 或提交参数异常会关闭绑定并记录 error，未执行事件计入 abandoned_events。

绑定继承创建者的 ExecutionScope。暂停允许轻量收件，但不执行 provider 或发布组件结果；普通 Signal 用户回调仍遵守原生命周期。关闭会退订、清理积压并取消在途请求，原 owner 等真实退休再释放资源。独立工具 scope 可以在 Stop 下运行。暂停单步可以跨宿主等待必需的物理或渲染结果，只向实际必要链授予执行许可，不把每次等待当成新的逻辑步骤；普通 Signal 回调不会借用这份许可。

`Signal.emit` 必须在引擎 owner 线程调用。它只表示事件发生，不声明生产资源已完成；生产与消费仍需真实完成关系。引擎不解释用户 RenderGraph 标签代表哪种业务图像。

| API | 当前行为 |
| --- | --- |
| `nn.export(model, path)` | 保存自含固定模型结构和参数的 `.inm` |
| `nn.save_checkpoint(model, optimizer, path)` | 保存参数、可选梯度及当前 SGD/Adam 状态 |
| `nn.load(path, device=...)` | 恢复固定模型，无需原作者 Python 类 |
| `nn.load_checkpoint(path, device=..., programs=...)` | 恢复模型和优化器；可借入已有预编译程序 |
| `finish_within_frames=...` 与 `budget=...` | 将上述保存或加载交给当前引擎共同执行管道，返回可观察请求；默认调用同步完成 |
| `nn.bake(model, path, optimizer=..., objective="mse")` | 冷构建并封装目标设备的预编译程序；附优化器时封装对应 mse、cross_entropy 或 scalar 训练程序，不预先固化逐帧调度表 |
| `program.bake(path, *sample_args, **sample_kwargs)` | 导出该已准备签名的模型、权重和原生程序；直接使用捕获的损失模式、ignore_index 和 reduction，不从默认目标猜测整数训练配方 |

INM 的文本 header 自含结构、层和张量约定，二进制 ckpt 区存数值载荷；资产 `.meta` 保存附加信息。当前编译产物针对准备时的设备和能力，不把一个目标的文件视为已经包含所有后端。已有 CPU/Vulkan Player 实测可运行预编译推理、训练和保存恢复；新增形状或算子仍需要重新构建相应程序。

正式模型、Torch 对照和学习示例维护在独立 Infernux 项目中。引擎仓库保留通用接口、此支持矩阵及必要回归。

## 策略分布

`nn.distributions.Categorical(probs=None, logits=None, validate_args=None)` 已提供 `probs`、`logits`、`mode`、`sample(sample_shape=())`、`log_prob(actions)` 和 `entropy()`。类别为最后一轴，前面的轴为 batch；抽样输出为 int64，形状为 sample_shape 加 batch_shape；log_prob 的动作输入可用 int32、uint32 或 int64，并与 batch 广播。策略网络与分布应组合在同一个 `Module.forward` 中，使梯度在同一静态图内传播；不能把两个独立编译模型自动接成一条反向图。离线、共同执行器和 INM/AOT 使用同一组原语。

`probs` 会按行归一化，取 log 前夹到 float32 的 `[eps, 1-eps]`；`logits` 使用稳定的 log_softmax。entropy 对对数概率使用有限下界，避免被屏蔽类别的 `0 × -inf`。这些定义参考 [PyTorch Categorical 实现](https://github.com/pytorch/pytorch/blob/main/torch/distributions/categorical.py)。`log_prob` 的反向只写浮点参数分支，整数动作没有梯度；越出类别支持域的整数动作输出 `-inf`，其选择分支梯度为零，不裁剪到其它动作。

目前要求可信输入：概率非负且有限、每行和大于零；logits 不含 NaN/+inf，每行至少一个有限值，可用 `-inf` 屏蔽类别。准备阶段检查结构、shape 和 dtype；尚不做运行时数值验证，默认 `validate_args=None` 与 False 使用该约定，True 明确报告未支持。不会为验证隐式读回整块 GPU 数据。完整 actor-critic 与场景仍待验收，已有采样与分布数学不等于完整 RL 支持。

`nn.rand(*size, device=..., dtype=..., requires_grad=...)` 当前生成 float32 的 `[0,1)` 张量，size 可为尺寸参数或 tuple/list；不支持空张量。模型捕获区内使用图设备且不新建 requires_grad 叶，区外可创建需要梯度的随机叶。`nn.manual_seed(seed)`、`nn.get_rng_state()`、`nn.set_rng_state(state)` 管理同一神经随机流；Linear/Conv2d 初始化也使用该流和同一个计算原语，不再单独调用 NumPy RNG。状态是拥有自己存储的 CPU uint32[4]，依次保存 64 位 seed 和下一个 64 位 counter 的低/高字。模型 checkpoint 保存模型及优化器，世界级 RNG 状态由应用显式保存恢复，加载一个模型不会重置其它模型的随机流。

内核采用 [Random123 的 Threefry2x32-20 算法](https://github.com/DEShawResearch/random123/blob/main/include/Random123/threefry.h)，每个输出使用独立计数器，高 24 位转为 f32。普通整数参数复用 CPU/Vulkan 的 signed int32 ABI，随机数直接在目标设备生成。一次实际模型执行原子保留所有随机节点的区间，跨帧切分、等待和重试复用已分配 key；prepare、Program 准备、bake 和模型加载只准备布局，不消耗随机数。执行开始前取消不推进，已经保留的区间在取消后仍被消耗，不回滚；多个请求按实际开始执行的先后取得区间。CPU 随机核的 16 字节标量存储计入共同容量，在提交 worker 前复查实际存储。

固定 `model.prepare_work()` 配方暂不支持随机层，它没有每次实例调用的随机状态绑定入口，会明确报告此限制；随机模型通过普通模型调用或 `nn.program` 在共同执行器中运行，预编译 INM 同样支持。不会在准备配方时消耗并永久冻结一批随机值。

Vulkan 的 int64 运算使用设备实际启用的 shaderInt64 能力；不具备该能力时明确报错，不转换为浮点数或改走 CPU。float32/int32/uint32 路径不请求该能力。离线编译器根据声明类型生成目标能力需求，不查询现场 GPU；预编译产物继续由运行设备验证。64 位张量的存储、任务长度、传输、已写区间和模型容量均按 8 字节元素计算，参数与梯度仍为 float32。参见 [Vulkan 特性规范](https://docs.vulkan.org/spec/latest/chapters/features.html)。

## 连续策略分布

交叉熵的输入是未归一化 logits。对于 `[N,C,H,W]`，整数目标形状为 `[N,H,W]`；对于单向量 `[C]`，目标为零维 int64。概率目标仍与 logits 同形。目标不接收梯度；mean 对非忽略位置的数量归一化，sum 直接求和。全部忽略时，mean 返回 NaN、梯度全零，sum 返回零、梯度全零。这与概率目标全部为零的语义不同。规则参考 [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)。

类别目标的合法数值为 `[0,C)` 或 ignore_index。执行器直接使用原设备上的 int64 数据，不展开 one-hot，也不为逐标签检查下载 GPU 数据。非法类别当前产生 NaN，不提供 Torch 的逐标签运行时异常；完整 64 位值参与比较，不截成 int32 或浮点。忽略位置的梯度直接写零。计数、求损失、反向和优化器都进入现有工作配方，按同一预算跨帧执行；不返回 loss 的训练也包含自己的有效数量计算。

需要预编译整数类别训练时，在正常 `program.prepare(inputs, labels)` 后调用 `program.bake(path, inputs, labels)`，让导出使用实际的目标布局和损失参数。`nn.bake(..., objective="cross_entropy")` 的无样本声明继续表示默认概率目标；两者共享模型容器、原生程序目录和加载管线。

`nn.exp/log/sqrt(input)` 与 `Tensor.exp/log/sqrt()` 使用共同的 CPU/Vulkan 计算原语和 VJP，支持受管执行、输入/参数梯度和预编译 INM。log 与 sqrt 不夹取输入；数学定义域和 IEEE 浮点边界保留原语义。

`nn.distributions.Normal(loc, scale, validate_args=None)` 提供 float32 独立标量 Gaussian：参数按普通广播规则形成 `batch_shape`，`event_shape=()`。支持 `loc/scale/mean/mode/stddev/variance`、`log_prob(value)`、`entropy()`、`sample(sample_shape=())` 和 `rsample(sample_shape=())`。多关节的联合 log probability 需要用户对关节轴求和，不会把 batch 轴暗中当作事件轴。

参数必须满足均值有限、标准差有限且大于零的可信输入约定。动态 `validate_args=True` 暂未实现，不为验证额外读回 GPU 张量。捕获内外都可以使用 Tensor 或标量参数；标量在捕获内形成真正的零维常量节点，随后按需广播。跨设备或跨捕获域的参数不能混合。

`sample()` 不记录对分布参数的梯度；`rsample()` 保留路径导数，形式为 `loc + scale * noise`。噪声通过两个现有 uniform 节点的 Box–Muller 变换生成，再明确停止梯度；使用 `log(1-U)`，不添加改变参数或密度的 epsilon。两个随机节点仍由一次 Run 原子保留计数区间，沿原神经图做预算、切分、传输与发布。prepare/bake 不抽取噪声；恢复随机状态可以重放同一设备上的采样。不同设备的超越函数精度不承诺逐位相同。

训练时在 `Module.forward` 内组合网络、分布和损失，再由 `nn.program` 提交。分布接口参考 [PyTorch Normal 源码](https://github.com/pytorch/pytorch/blob/main/torch/distributions/normal.py)。这提供连续策略的分布数学，PPO、场景动作、奖励、reset 和实际学习仍由后续闭环验收。

## 截断损失与逐元素比较

`nn.minimum/maximum(input, other)` 和同名 Tensor 方法执行 float32 逐元素比较，使用普通广播规则。两值相等时，两侧各得到一半上游梯度；同一 Tensor 同时作为两侧输入时，贡献按原规则相加。NaN 输入传播到输出，严格落选侧的梯度直接写零，避免乘零造成的 NaN/inf 污染。规则参照 [PyTorch 的 minimum/maximum 导数定义](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)。

`nn.clamp(input, min=None, max=None)` 和 `Tensor.clamp` 当前支持有限、可表示为 float32 的标量边界，至少指定一侧；Tensor 边界和 out 尚未支持。输入恰好等于边界时保留输入梯度；反向边界使有限输入输出 max、输入梯度为零。此处采用 [PyTorch 标量 clamp 的 inclusive 子梯度](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/FunctionsManual.cpp)，与 Tensor 边界的梯度分配区别开，不以两次 minimum/maximum 代替。

`nn.ELU(alpha=1.0, inplace=False)` 与 `nn.functional.elu` 使用同一原语。alpha 是有限 float32 标量，零点导数取 alpha，负分支在原点附近用局部展开避免 exp(x)-1 相消。反向先形成局部导数，再乘上游梯度；输入 NaN 显式传播，不依赖 GLSL 扩展 Exp 对 NaN 的行为。inplace 暂未支持。

这些操作沿原 Plan、VJP、共同执行器和 INM/AOT 工作。桌面 NNNP1 的 `PPOObjective` 与 Torch 参照演示概率比截断、价值截断、entropy 项及 rollout 数据停止梯度；该目标函数本身还不是完整 PPO 训练器，rollout/GAE、更新循环和实际环境学习另行验收。

`nn.utils.clip_grad_norm_` 使用与 [Torch 梯度裁剪](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/utils/clip_grad.py) 相同的 `min(max_norm / (norm + 1e-6), 1)` 缩放公式。max_norm 当前要求有限、非负、可表示为 float32。归约先在最多 256 个元素内取幅度，再缩放求平方和并逐级合并，避免大数直接平方造成的额外溢出；真实范数超过 float32 范围时仍返回 inf。非有限值按浮点计算传播，error_if_nonfinite=True 暂不支持，不隐式读回 GPU 验证。

离线调用返回同设备零维 Tensor；查看 item() 或 cpu() 才是明确的读回。梯度存储遵循本库的快照语义：完整计算后一起替换 `.grad`，旧梯度 Tensor 仍指向原值。全部梯度为 None 时返回零，并保持 None。裁剪本身不修改模型参数版本或 Adam 计数。

在 `nn.program` 中，调用顺序为 `zero_grad → model → loss.backward → clip_grad_norm_(model.parameters(), max_norm) → optimizer.step`。目前一次裁剪完整模型域；范数代理不能返回、用于条件判断或进一步运算，程序仍可返回 `loss.detach()` 或 None。准备阶段只编译；受管运行在原 update 游标中执行归约、缩放和更新，成本、容量及预编译程序包含这些操作。裁后梯度、参数与优化器状态在最终发布时一起生效；中途取消保留已完成 backward 的梯度，不发表部分裁剪结果。

`program.is_prepared(*args)` 按 Program 的完整输入签名查询已捕获配方及当前模型结构资格，不执行采样、IO 或编译。shape、device、dtype、requires_grad 都属于签名；train/eval 或 recompile 使旧图失效。作者工具可据此调用 prepare，避免另记一份仅含 shape 的准备状态。该查询不代替后端实际准备完成的提交条件。

桌面 NNNP1 的 GaussianPPO 提供采样、GAE 和多轮 minibatch 更新示例。更新开始时固定行为数据，old log probability/value 在各 epoch 保持不变；每次成功优化只推进一次参数发布计数，并检查整轮更新谱系。动作维数由环境传给训练器，不依赖模型作者类的额外属性。单个预编译 INM 仍只含其导出的输入签名和捕获模式，训练图与部署图分别准备、保存。

共同执行器的历史 host 成本是条件上估。画像超出宿主预算时，请求显示无法建立完成预约，仍保留原管道中的合法校准机会，不把该估计当作必然无法执行的硬下界。已有预约、前景优先、实际额度扣费和原截止窗口继续生效；这不构成硬实时完成保证。
