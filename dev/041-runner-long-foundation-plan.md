# 041：Runner Long 迁移基础、权威语义与扩展能力执行计划

> 2026-09-15 PC 主链复核：Windows Release CTest **90/90** 通过（含 RenderTexture、UI Transform、GPU buffer/mesh、compute physics exchange、DPI policy 与场景 residency）；GPU 计算专项 **4/4** 通过；真实 `Infernux041Lab` MCP `core` 捕获仍可用。`input` 复核已越过运行期事务错误，但 Space 物理动作仍未触发，保留为输入时序缺口，未伪造通过。计划清单当前 **104/257 + 37/76 = 141/333（42.3%）**；本轮只更新有证据的统计，不把局部测试冒充 A03.4、Taichi 300 FPS 或完整跨平台收口。

> 2026-09-14 屏幕 UI 原生姿态段：控件几何范围绑定 Transform，在原生发布阶段应用位置/Z 旋转和 Scale 增量，并使用不可变本地顶点快照避免跨帧累积漂移；姿态变化不再重建 CPU ImGui 几何，布局、裁剪、材质、文字和绘制顺序保持不变。Windows Release 构建成功，针对性原生测试 **2/2**、Python UI 回归 **152通过／1跳过**。整帧/可见编辑器验证、尺寸变更、GPU 上传粒度、跨平台测量和其余041清单仍待推进，严格完成度以当前清单复算为 **140/333（42.0%）**。

> 2026-09-14 屏幕 UI 批量动画审计：在 1000 个 Canvas 控件、64 材质、逐控件每帧变换的真实负载下，屏幕位置 **102.48275 ms P50**、旋转 **100.1121 ms P50**，每帧提取 1000 个控件。此前只移动 Canvas owner 的 0.32695 ms 夹具未改变子控件几何，已明确排除。世界 UI 已有原生姿态段，屏幕 UI 的位置/旋转仍写入 CPU ImGui 顶点；需要屏幕侧原生姿态分段或等价的布局/姿态拆分，才能继续优化，未新增完成勾选。详见[执行记录](041-execution-log.md)。

> 2026-09-14 世界 UI 批量姿态更新：局部几何绑定原生场景 Transform，在提交时读取位置/旋转/图层，移动不再重新生成文字和形状；保留父级影响、忽略自身 UI Scale、代际生命周期和显式矩阵 API。1000 世界标签随父级旋转，Python 提交 P50 **57.54945→0.3992 ms**，提取 **1000→0**；独立原生 A/B 的几何构建 **0.8775→0.134 ms**。这些是阶段数据，不代表整帧 FPS。真实 Vulkan 回读确认缓存/新绘制一致、图层和帧内姿态正确、删除后不会误绑定复用槽；完整原生 **90/90**、Python **6352通过／12跳过**。可见编辑器的字体、材质、屏幕布局点击及移动旋转后的世界按钮均通过。屏幕 UI 批量动画、布局/内容失效、GPU 上传、整帧/跨平台和世界 MCP 自动发现仍未收口；严格总进度 **139/332（41.9%）** 不变。详见[执行记录](041-execution-log.md)。

> 2026-09-14 UI 局部 Transform 更新：原生依赖快照提供本次变化的控件索引，屏幕 UI 按父子依赖传播，世界 UI 按实际世界姿态与图层更新；移除局部变换导致整组几何缓存失效的路径。1000 控件中只移动一个，提取数量 **1000→1**；同版本恢复旧失效策略作 A/B，Python 提交 P50：**屏幕 92.17135→0.44045 ms，世界 57.95375→0.40970 ms**，旋转亦约 0.4 ms。这里只计稀疏修改的 CPU 提交，不是整帧 FPS，也不代表全部控件同时动画的成本。最终原生 **90/90**、Python **6351 通过／11 跳过**；真实编辑器已验证独立移动/旋转、父级移动后屏幕点击、世界面板移动旋转后点击、字体和材质更新，场景恢复干净 Edit，无遗留测试进程。尺寸/布局失效粒度、大批量动画、GPU 上传与整帧/跨平台矩阵、世界按钮 MCP 自动发现仍待推进。严格清单 **139/332（41.9%）** 不变，完整 UI 与041尚未收口。详见[执行记录](041-execution-log.md)。

> 2026-09-14 UI 增量提交：控件变更直接通知各渲染目标的几何缓存组，移除逐控件测量/缓存键轮询及已无用途的版本计数器；材质变更只通知实际使用者，布局和字体仍按依赖刷新，自定义绘制保持原有顺序。1000 控件、64 材质、单文字变化的 Python 提交 P50：**屏幕 1.0631→0.4457 ms，世界 0.8377→0.4165 ms**；共享材质变化为屏幕 1.5343／世界 1.48655 ms，仍只重建 16 个使用者。数字不含原生与 GPU 时间，不代表整帧 FPS。最终完整 Python **6341 通过／11 跳过**；真实编辑器的材质首次改色、屏幕布局点击、世界按钮回调、字体和父级变换均已回归，03 恢复干净 Edit，无遗留测试进程。严格清单仍为 **139/332（41.9%）**；局部 Transform／布局失效范围、完整负载与跨平台验证、世界按钮 MCP 自动发现仍未收口，不新增整项勾选。详见[执行记录](041-execution-log.md)。

> 2026-09-14 UI 提交继续压缩：材质/贴图绑定变更与文字/颜色/悬停变更分开，保留原引用的实际owner/版本检查，但不再每次扫描所有序列化字段；未变化控件也不重复解析渲染器类型。1000控件/64材质、单文字变化的Python提交P50 **屏幕2.7524→1.0631ms、世界2.2957→0.8377ms**；共享材质变化仍只重提取16个消费者，屏幕3.0791→2.0850ms、世界2.3566→1.8084ms。这些不含原生/GPU阶段，不代表整帧FPS。新增绑定/序列化恢复/渲染器热替换等16例，全量Python **6333通过/11跳过**；真实编辑器材质首次改色、三种屏幕布局点击、世界按钮回调、中文/字体/父级变换回归通过，Console无新增错误，场景恢复干净Edit。没有增加fallback或校验链；GPU上传粒度与完整UI/多平台仍未收口，严格总进度139/332不变。

> 2026-09-14 UI 原生几何缓存：移除Python绘制参数回放，改为只为变化控件生成原生不可变几何包、按原有顺序批量提交；未变化控件复用顶点/索引/文字几何，保留世界深度/图层、材质、裁剪、字体失效和自定义绘制顺序。1000动态文字的原生构建P50 **屏幕0.5067→0.0391ms，世界0.5287→0.0579ms**。1000控件/64材质的Python提交仍约屏幕2.75—2.76ms、世界2.19—2.30ms，但单字变化只提取1个控件并批量提交1次；共享材质变化只提取16个消费者。这里分别是原生构建和Python阶段数据，**不能相加推算整帧FPS**。最终原生90/90、Python6317通过/11跳过；可见编辑器MCP验证首次材质改色、中文/字体/父节点变换、三种屏幕布局真实点击和世界按钮回调均通过，无新增错误；03恢复干净Edit，Hub已重开。Python大规模遍历、GPU上传粒度、跨平台与完整UI性能仍待继续；严格完成度仍139/332，未提交/发布。下方保留前序阶段数据作对照。

> 本轮最终验证：**原生90/90、Python6307通过/11跳过**。已重开真实编辑器，用MCP验证首次共享材质改色在屏幕/世界文字及世界按钮文字即时显示；横排、扩宽Fill、竖排三个屏幕按钮位置和世界按钮真实点击均触发0→0.75回调。Console零警告/错误，临时材质通过可撤销删除清理，03场景恢复干净Edit；未提交或发布。完整UI性能与041仍未收口，严格总进度维持139/332。

> 2026-09-14 UI 材质密集路径续进：共享材质只在一次发布中检查一次版本，材质/RenderTexture更新仅重新提取对应控件的绘制参数，不再清空全部控件缓存。1000控件、64材质、每帧修改一个共享材质：重提取1000→16个，隔离CPU提交约62ms→**屏幕3.208ms / 世界2.817ms**；单个文字/颜色变化仅重提取1个，屏幕2.976—3.008ms、世界2.541—2.612ms。实机发现的首次材质改色被默认白色覆盖也已定位：创建模板把Color写成Float4，管线同步误丢弃兼容数值；现模板使用Color，原生同步保留同形vec4并采用着色器语义。原生全量90/90通过，完整Python与最终可见编辑器回归见执行记录。**数据不含原生细分/上传/GPU时间，不代表整帧FPS；1000次逐控件提交、动态文字几何、跨平台仍待继续，不增加整项勾选。**

> UI查询最终复测：完整Python **6297通过/11跳过**；100/300控件隔离CPU布局 **2.3796/7.0758ms**，各只进行100/300次期望尺寸计算。真实编辑器已恢复03干净Edit；仍继续密集材质/纹理、动态几何和跨平台矩阵，不将局部CPU收益换算为整帧FPS。

> 2026-09-14 UI 查询续进：每次矩形读取复用已校验的 owner、Canvas 和 Transform，移除字段读取时反复构造的类型集合。300控件重排中的 owner 查询2106→605次；中间隔离测量100/300控件2.6407/7.2156ms，对比上轮4.2650/12.4507ms。真实Canvas测试也发现尺寸编辑遗漏Transform联合撤销，现由既有属性事务一次提交尺寸与位置；屏幕/世界、0/37/90度、Undo/Redo六组覆盖，Inspector/布局105项通过。可见编辑器已更新，字体/父级变化及屏幕、世界按钮回调通过。最终全量与隔离结果见执行记录；这些是局部CPU阶段，**不代表全部UI或041完成，也不增加整项勾选**。

> 2026-09-14 自动布局续进：UIFrame 原来查询每个子控件都重算全部兄弟，300控件单次重排约1.12秒、13.5万次期望尺寸计算。现在一次布局生成整组结果，复用嵌套Hug测量，并省去固定尺寸父节点的无效子树测量；100/300控件CPU布局从128.46/1123.47ms降至4.13/12.06ms，尺寸计算为100/300次。**仅是布局微基准，不是整帧，也不代表剩余12ms可以验收。** 新增14组回归，覆盖布局/文字尺寸/生命周期与世界UI不进入屏幕排列；真实编辑器验证横排、扩宽+Fill、竖排以及按钮在三个新位置的回调。完整Python与最终隔离复测见执行记录。逐控件几何更新、剩余矩形/Transform提取、多资源与跨平台仍待推进，完整UI/041保持未收口。

> 自动布局最终复测：Python完整6289通过/11跳过；100/300控件隔离CPU布局4.2650/12.4507ms，仍各只测量100/300个尺寸。可见编辑器场景恢复干净Edit，无遗留测试进程；完整UI与041仍待后续工作。

> 2026-09-14 UI 文字主路径续进：修复已加载字体仍重复访问文件系统的问题；在一次布局结果中复用字符解析、字体选择和字宽，绘制不再逐字重算，同时修复文件字体缓冲的所有权泄漏。1000 个显式文件字体标签、每轮重新布局并生成顶点的 CPU P50：拉丁文字463.22→2.40ms，中英混排747.23→3.26ms；巨大基线包含重复文件路径解析，**不能套用到默认字体、GPU或整帧FPS**。960组文字几何、相对路径/CWD、字体重载与分配释放检查通过；Windows完整原生90/90通过。真实03场景冷开、屏幕/世界字体实时切换、父级变换及世界按钮回调通过。最终全量Python和隔离复测见执行记录。继续频繁重排、逐控件原生几何更新与多资源UI矩阵，**不增加041整项勾选**。

> 本轮文字最终复测：完整Python6275通过/11跳过，原生90/90通过；隔离文字基准2.3867/3.2540ms（拉丁/中英混排，布局+顶点生成）。真实03世界按钮回调、12六个屏幕按钮和字体热切换均通过；编辑器已恢复03干净编辑状态，无遗留测试进程。全部UI负载与多平台仍未收口。

> 2026-09-14 UI 原生渲染续进：静态几何不再每帧/每相机重复转换和上传，GPU 缓冲改为复用 VkCore 的实际帧槽；世界 UI 在相机透明排序后合并兼容的连续绘制区间，保留裁剪、材质纹理、深度、图层与混合顺序。1000同字体文字的原生CPU录制P50，屏幕约0.09→0.01—0.02ms，世界2.65→0.026ms；此世界用例可以合成1次绘制，**不是全部世界UI场景、GPU耗时或整帧FPS的保证**。新增真实Vulkan像素回读与四帧槽隔离测试，完整原生90/90通过；03/12真实可见编辑器的冷开、文字/父级变换、世界按钮回调和六个屏幕按钮回归通过。Python全量与最终隔离复测见执行日志。频繁重排、不同纹理的大批控件、原生文字细分及完整多平台矩阵仍待推进，**不增加041整项勾选**。

> 2026-09-14 UI 动态提交续进：统一内置控件绘制参数缓存，移除文字/图片/按钮各自重复的依赖读取；只改一个控件时复用其余控件参数，保留原生绘制顺序和批处理。1000个真实UIText、每帧只改一个颜色/文本的Python提交P50：屏幕56.4—56.9→2.05—2.13ms，世界47.7—48.9→1.61—1.66ms；每次只重新提取1个控件，仍调用1000次原生AddText，**不是GPU时间或整帧FPS**。补齐FixedSize文字不牵动无关布局、父级裁剪失效，以及原生子节点间换父级遗漏层级版本通知的问题。Windows原生89/89、完整Python6275通过/11跳过；可见编辑器03/12冷开、文字/父级更新、六个屏幕按钮和真实世界按钮回调均通过，未保存测试改动。原生几何/上传、密集材质、频繁布局变化及跨平台完整UI性能仍待继续，**不增加041整项勾选**。详见执行记录。

> 2026-09-14 世界 UI 输入续进：运行时指针与 Scene 点选共用原生批量射线投影，保留拖拽越界坐标、Transform/父级旋转、深度遮挡和图层语义。1000世界控件全未命中的输入基准16.38→0.570ms；新增重叠压力测试又推动事件查询按表面优先级从前到后命中即止，避免继续遍历后方控件。真实03场景通过MCP绑定并点击世界按钮，确认目标值0→0.75，Stop/重开丢弃临时修改；12场景六个屏幕按钮同样通过。原生89/89通过；最终顺序优化后的全量Python与隔离性能复测见执行日志。**本轮仅推进输入链路，动态布局和逐控件绘制更新仍待优化，不增加041整项勾选。**

> 本轮最终复测：Python6239通过/11跳过，原生89/89通过。1000控件输入映射+命中P50：屏幕0.0624ms、世界0.5757ms；包含发现/投影/事件处理的完整CPU输入阶段：屏幕0.0720ms（全未命中）/0.0718ms（全重叠），世界0.7240/0.6334ms；世界重叠在按优先级查询前为16.3935ms。均不等于绘制或整帧耗时。12场景最终三段240帧平均0.950—0.979ms、P95约1.64ms，GUI有既有空闲帧不重建行为，不能把整帧差异全部归因于本轮。当前编辑器与Hub已恢复，没有遗留测试进程。完整UI/041仍未收口。

> 2026-09-14 UI 输入专项：屏幕 Canvas 已复用按顺序排列的命中包围盒和裁剪区域，颜色/透明度动画不再触发命中几何重建；布局、原生 Transform、交互策略和场景身份仍分别参与失效。1000 个控件全未命中的输入微基准从28.99ms降到0.0622ms，世界目标提前拒绝越界点后从26.73ms降到16.38ms（中间轮18.89ms），后者仍不可验收。同步修复父级移动时渲染侧子布局旧缓存，以及静止鼠标下 Hover 不刷新的问题。真实03/12场景回归和完整Python6230 passed/11 skipped通过。最终12场景三段240帧平均1.038/1.046/1.110ms、P95约1.98ms，保持1080p、4×AA、13个Pass；仅是本轮场景实测，不将整帧收益全部归因于本次修改。下一步继续世界输入批处理、动态布局/逐控件命令复用和整帧矩阵，**完整 UI 和041仍未收口，不增加整项勾选**。

> 2026-09-14 UI 第二轮：几何依赖已改为原生批量快照，空材质槽不再逐帧进入 Python 解析。原生生命周期句柄保护对象销毁/重用，结构与视觉变更仍使用既有版本，不添加重试、兜底或哈希验证链。1000 控件依赖检查 CPU 微基准：屏幕静态13.390→0.0119ms，世界静态15.277→0.0028ms；其它3D物体每帧移动并更新矩阵时，世界15.370→0.1279—0.1283ms。最终原生89/89、完整Python6221 passed/11 skipped通过。真实编辑器03场景冷开即可显示，屏幕/世界文字修改与世界父节点位移旋转正确，12场景六按钮及Stop/重开通过，Console无警告/错误。12场景两组240帧窗口平均1.881/2.185/1.903ms和1.682/1.219/1.197ms，存在明显波动，不能把依赖微基准当作整帧收益。这里仅推进依赖检查，**布局、输入、材质密集场景、逐控件命令更新和整帧性能尚未完成，不增加整项勾选**；旧阶段数据保留用于对照。

> 2026-09-13 用户最新验收决议：**描边渲染主路径暂记成功，600 FPS 已可接受；1000 FPS 不再作为本项收口阻塞。** 下方原性能目标与未验收声明保留为历史。下一项优先专项优化屏幕空间与世界空间 UI：分别测量布局/文字、输入命中、内容失效与原生提交；保留同帧变化、交互、深度与多 Camera 语义，不通过降低刷新频率或删减控件提速。完整 041 尚未完成，现有清单勾选不因这项阶段决议自动增加。

> 2026-09-13 最新 Outline 回归：维持1080p、4×逐采样描边与13个项目Pass，通用Vulkan相邻同队列批量提交把驱动调用从8次/帧降到2次，保留跨队列/上传/完成栅栏合同。同时修复MCP一次UI快照错误开启持续采集的问题；六按钮操作之后持续采集仍关闭，快照发布计数不再逐帧增加。最终可见编辑器三段240帧平均1.301—1.317ms（760—768FPS），P95约2.07—2.11ms；原生88/88、Compute/JIT126项、MCP88项通过。**1000FPS目标仍未达成，视觉验收仍待用户确认，完成度保持139/332（41.9%）**。详细证据见执行日志；未降低画质或修改场景资产。

> Outline AA 补充验收：用户已明确否决“主颜色/深度4×、描边仍锯齿”的结果。公共 Texture2DMS/UInt、反射、原始多采样读取及逐样本合成现已接通，真实 Vulkan ID/深度/颜色回读测试通过；Scene12 的逐采样描边与六按钮交互已在可见编辑器校验，无校验错误。最终原生88/88、Python6207通过/11跳过；无校验层整帧实测1.428—1.628ms（614—700FPS），P95 2.54—2.84ms，尚未达到1ms目标，待继续优化及用户视觉确认，不新增整项勾选。保持对象ID/flags离散身份，不能平均ID，不能将单独模糊滤镜当作4×MSAA完成。此前823—840FPS属于旧mask1版本，不代表新逐样本版本性能。最终目标仍为1080p、4×、全部效果下真实整帧接近1ms，不添加场景私有引擎分支。

> 2026-09-13 Outline 性能验收更新：`12_OutlineOwners` 在当前 1920×1080、4×MSAA 下，真实可见编辑器整帧目标接近 1000 FPS（约 1 ms），保留描边、遮挡斜线、GPU 变形、世界 UI、交互与离屏相机。以多个稳态窗口的平均/P95 帧时和 GPU 时间验证，不用瞬时 FPS、game-only 估算、降分辨率或关闭效果代替。现有 MSAA1 基线不满足该目标，尚未验收；先优化通用 UI/绑定/渲染提交，再补齐多采样颜色、深度与 owner 数据合同。

> 2026-09-13 追加：GPU 编译器活动设备/LLVM 依赖进一步裁剪，最终88/88原生与126项Compute/CPUJIT定向测试通过，子仓库安装目录CI已修正。总进度仍139/332（41.9%）。新增性能复核：Outline在1920×1080、MSAA1时，验证层开启约5.6ms，关闭后约3.5—3.6ms；后者CPU管线准备仍约2.2ms，不能视作性能收口，继续定位通用渲染提交开销。没有降低画质或修改场景来提高FPS。

> 2026-09-13 最新复核：主计划 **103/256**、Compute **36/76**，合计 **139/332（41.9%）**，**193 项待验收**。GPU 编译上下文已简化为直接输出 SPIR-V/元数据，移除重复输出层级和序列化；ComputeBuffer/Kernel/Readback 宿主服务生命周期与退出顺序已修复。MCP 的保存基线、临时文件资产身份、严格场景校验及文件监听/发布时序通过独立复核和 **185/185** 定向测试。最终完整原生 **88/88**（16 项 Vulkan，129.91 秒）、Python **6199 passed / 11 skipped**（285.31 秒）；正常更新 Lab 的 MCP 后，真实编辑器中的雪面、早期软体及 800×450 RenderTexture 可见回归通过，正常退出无验证错误/遗留测试进程。项目玩法资产未重写；缓存消费回归不等于冷编译或 300 FPS 数量阶梯验收。完整范围粗估仍为 **12—18 个单人全职工程周**；原生依赖深裁、编译服务/ABI、发行、数量阶梯性能与跨平台仍待完成，B05 不提前勾选。

前序阶段记录：真实可见的 `12_OutlineOwners` 已通过同一 Rigidbody 零件分组、RGBA32F 数据图与独立深度、红白边界、主动/遮挡斜线和显式深度遮挡。6次真实按钮点击、16种状态/投影组合及4种完全遮挡组合均通过，GPU 变形与旋转跟随，图拓扑构建次数保持2；冷启动直接出图，Stop/重开/正常退出均无错误。当时原生 **83/83**（Khronos 验证层）、项目真实原生登记测试 **11/11**、完整 Python **6110 passed / 13 skipped**、20个教程生成检查通过。两项额外 skip 来自 E: 测试目录不支持8.3短路径，已逐项核对；完整记录见 `041-execution-log.md`。当时尚未完成的世界文字和细遮挡验收现见上段及 A10，以下段落只保留阶段历史。

> 2026-09-13 A09→A10 历史复核：主计划 **94/256**、Compute **36/76**，合计 **130/332（39.2%）**，**202 项待验收**。

本轮完成 Windows 纵向验证：公共 Fullscreen Pass 支持深度比较/写入、片段深度输出和非预乘 alpha 混合；修复独立深度图被错误别名到 Camera 深度、Compute 上传使用无效队列阶段，以及同源内嵌材质因路径身份冲突而遗留 GPU 资源的问题。复用现有 RHI、提交票据和运行时材质 ID，移除错误分支，不增加恢复管理器。Shader 输入布局只在管线创建边界核对，避免作者热更产生未绑定描述符的 Draw。Release 构建、开启 Khronos 验证层的 **83/83 CTest**、**6112 passed / 11 skipped** Python 回归均通过；完整原生日志无验证错误。真实 041Lab 11 号场景通过 GPU 变形、子网格/选择/颜色变更、六次按钮输入、Stop/重开及正常退出，Console 与原生验证日志无错误，退出无遗留进程；目前保留一个正常编辑器在 clean Edit。独立深度的其它平台限制与完整 View/MSAA 边界、A10 分组 ID/eye depth/状态和原项目合成仍未收口，因此不新增整项勾选。完整证据与已知边界见 `041-execution-log.md`。

> 2026-09-13 Renderer 集合续进：主计划 **94/256**、Compute **36/76**，合计 **130/332（39.2%）**，剩余 **202 项**；完整范围仍估计 **12—18 个单人全职工程周**。本轮只新增勾选 A09 的对象集合/submesh/替代材质/逐 draw 参数条目：`RendererSelection` 接入现有 RenderGraph 和当帧 RenderWorld 几何，复用身份代际、参数捕获、GPU 顶点缓冲及批处理，不复制网格、不新建渲染路径。真实可见 `11_SelectionStudies` 已验证 GPU 变形跟随、两个 submesh 独立颜色、遮挡、清空/恢复选择及 Play→Stop→重开；9次选择变更没有重建 Python 图拓扑。Windows 1920×1080、240帧窗口平均1.128ms/P95 1.838ms，仅为该演示的编辑器帧时，不代表 GPU timestamp 或软体验收。无 RenderStack 时新建管线的目录刷新也已通过真实编辑器验证。最终原生 **83/83**、完整 Python **6089 passed /11 skipped**、20个教程生成检查及空白检查通过；新编辑器中公共 API 脚本热更、6次真实按钮点击、Stop/重开均通过，Console0警告/0错误，11号场景留在 clean Edit。完整 A09/A10、发布与多平台仍未收口；以下旧状态只作历史证据。

> 2026-09-13 最新续进：主计划 **93/256**、Compute **36/76**，合计 **129/332（38.9%）**，剩余 **203 项**；完整范围预估 **12—18 个单人全职工程周**，按验收清单而非代码量统计。A09 已新增真实可见的 `10_PlanarReflectionGallery`：文件材质、Inspector 引用、编辑模式反射预览、环绕/裁剪按钮、镜面平移/倾斜，以及两次 Play→禁用/启用→Stop→重开已验证，Console 0 warning/0 error。引擎统一 NumPy 矩阵桥接、补齐编辑模式 LateUpdate，并修复 native-only 快照遗漏及场景替换引用失效；删除字段的重复缓存捷径和多余清缓存，不增加哈希/恢复链。最新完整 Python **6077 passed /11 skipped**，原生 **83/83（15项真实 Vulkan）**，20个教程生成检查及空白检查通过。此前 RenderTexture 的正式 Windows Player 验收仍有效，但新展厅未重新导出；完整 View/资源合同、多平台、性能矩阵及其余041条目仍待完成，不据此增加整项勾选。下一段优先推进 A08/A09 的 Renderer 集合/submesh/逐 draw 公共接口，服务 A10 原项目描边消费者。历史状态仅作阶段证据，详细失败与修复见[执行记录](041-execution-log.md)。

> 2026-09-13 材质 RenderTexture 作者链续进：严格清单为主计划 **93/256（36.3%）**、Compute **36/76（47.4%）**，合计 **129/332（38.9%）**，剩余 **203 项**。完整范围估计 **12—18 个单人全职工程周**，不是自然日承诺。本轮材质纹理槽支持导入的 RenderTexture，并与 Camera、运行时材质副本、UIImage 材质消费共享 GPU owner。可见 041Lab 已通过 Project 拖拽保存、场景重开、Cube/屏幕 UI 同时取样、333×187→517×291/HDR/4×重导入、删除后 Undo 重连；恢复原09场景 clean Edit、Console 0 warning/0 error。只在资产变更时刷新绑定，不加逐帧哈希/目录轮询。原生完整 **83/83（15项真实 Vulkan）**；Python 全量 **6039 passed /11 skipped**，最后引用互通改动定向 **218 passed /1 skipped**，20 个生成教程及改动空白检查通过。**UIImage 直接纹理槽持久化、最终 Player/其它平台及 A09 剩余矩阵未完成，不新增整项勾选。** 下方旧状态仅作历史证据，详细结果见 [执行记录](041-execution-log.md)。

> 2026-09-13 资产链路/相机生命周期续进复核：严格清单仍为主计划 **93/256（36.3%）**、Compute **36/76（47.4%）**，合计 **129/332（38.9%）**，剩余 **203 项**。本轮补齐 RenderTexture 的原生描述导入、同 GUID 重导入、Library 二进制制品及项目/已选插件的 Cook 收集；修复同帧相机缓存失效与主相机删除后的悬空引用。Windows 完整重建后，原生 **83/83**（含15项真实 Vulkan）、Python **6004 passed /13 skipped** 全部通过。真实可见041Lab经 MCP创建/修改/删除资源，二进制描述同步更新，原09场景保持 clean Edit，Console 0 warning/0 error；引擎侧完整窗口截图已人工目视核对。**资源 Inspector、持久 Camera/材质/UI 引用和最终 Player 端使用尚未完成，因此不新增整项勾选。** 完整范围仍估计 **12—18 个单人全职工程周**（桌面核心约8—12周，包含在前者内），不是自然日承诺；下方历史数字只作阶段证据。详细失败、修复与最终回归见 [执行记录](041-execution-log.md)。

> 2026-09-13 相对 RenderTexture 续进复核：主计划 **93/256（36.3%）**，Compute **36/76（47.4%）**，合计 **129/332（38.9%）**，剩余 **203 项**。本次只新增勾选 A09 的资源描述/相对尺寸条目：Game 像素参考、Camera/材质/UI 同 owner 更新、失败不部分发布已形成测试证据；持久资产编辑、导出及其它平台仍单独待验收。Windows Release 完整原生 **82/82**、全量 Python **5963 passed / 11 skipped**，随后文档/资源/View/启动回归 **101 passed**；真实编辑器经 MCP 切换 Game 1920×1080→1280×720→2560×1440→1920×1080，半尺寸目标自动变化、固定目标不变、隐藏 Game 后仍出图，Console 0 warning/0 error。完整范围继续按 **12—18 个单人全职工程周**规划，桌面核心约 8—12 周；不是承诺自然日交付日期，也不是按代码行数估算。

> 2026-09-13 续进复核：严格清单仍为 **128/332（38.6%）**，剩余 **204 项**，完整范围工期估计仍为 **12—18 个单人全职工程周**。A09 通用 history 和按 Camera 输出采样配置构图/缓存已有 Windows 证据；TAA 的 motion/depth MSAA 不匹配已修复，真实编辑器双目标、resize、4×→1×→4× 和效果开关/参数更新通过。全量 Python 回归从13项失败修至 **5946 passed / 11 skipped**，随后中英文示例及输出采样定向20项通过，最终可见编辑器回归 Console 0 warning/0 error。完整资源调度、导出及多平台仍未收口，不能把局部实现计作整项验收。详见 A09 最新记录和执行日志。

> 2026-09-12 最新清单复算：主计划 `92/256`（35.9%），Compute 附录 `36/76`（47.4%），合计 `128/332`（38.6%），尚有 `204` 个未勾选项。本行表示当前完整验收清单进度，不是按代码量计算；此前 **85%—87%** 等局部工程估算不能用作整个 041 的完成度。按范围不再扩张、单人全职等效工作量，完整 041 暂按 **12—18 周**规划，桌面核心纵向交付约 **8—12 周**；Android/Web JIT 路线、性能矩阵及设备验收仍可能拉长工期。后续历史段落仅保留各阶段证据，不覆盖此处最新数字。

> 2026-09-12 当前复核：主计划明确勾选 `76/255`（29.8%），Compute 附录明确勾选 `36/76`（47.4%），合计 `112/331`（33.8%）。严格计数仍要求整项、多消费者和多平台证据后才勾选。当前可复用工程实现约 **77%—79%**，最终用户验收约 **54%—56%**。A17 的 10,000-ray Windows Release 基线 1.736 ms 明确不通过，已经移入后续物理查询管线级优化；本轮转向 A06，Canvas-free 世界 UI 已贯通普通 Transform、相机 ViewProjection、Scene/Game RenderGraph、深度遮挡、世界射线到局部二维坐标的事件输入和旋转面板局部裁剪。真实 041Lab 对同一旋转面板开关裁剪作了 Game/Scene render-target 对照，并证明 Game View 的真实鼠标能命中旋转世界按钮、驱动 Hover 状态；输入与渲染现共用 `effective_game_camera`，射线显式使用 Game 目标尺寸，不再依赖 Camera 最近一次被其它视图写入的尺寸。原生 `RenderWorld` 进一步把每个世界 UI 根作为一个透明排序单元：同一面板保持作者命令顺序，不同面板在每台 Camera 实际绘制时按其裁剪空间深度背到前稳定排序。041Lab 两个重叠半透明面板在不改变层级顺序的情况下交换空间深度，Game render-target 随近面由红切蓝而对应切换混合主色，且继续受场景深度遮挡。Slider 已改为读取事件携带的统一输入表面尺寸，不再反向依赖屏幕 Canvas 字段；真实 Game View 按下后越界拖动把世界 Slider 从 0 连续推到 100，证明捕获仍归原世界表面。Frame/Image 的统一提交测试和真实 Text/Button/Slider 输入出图也确认普通场景物体直接承载既有 UI 组件，不需要另一套世界 Canvas。文字布局现把作者 `font_size` 定义为 Figma/CSS 的 em 逻辑像素，并从每个字体自身的 SFNT `head/hhea` 表推导 stb 栅格换算；PingFangSC 同字串的 22/32/48/72px 可见宽度与浏览器参考仅差 0—1px。进一步的显式三行、中文/Latin 固定宽度换行、4px 字距和越界裁剪实测与 CSS 参考一致；`TextOverflow.Clip` 也已接入编辑器与运行时共享渲染主干。Roboto-Medium 32px 在浏览器和引擎的可见宽度同为 431px，证明换算按字体自身度量工作；DPI atlas 重建也会同步清空自定义字体缓存。UIText 的 Auto Width/Auto Height 现由一次预布局生成瞬态逻辑尺寸，UI Editor、Inspector 和 Player 不再分别回写作者 `width/height`；绘制、Frame 排版、裁剪和 raycast 共用该矩形，Canvas 缩放测量会归一回逻辑像素。真实 041Lab 同时显示固定宽、Auto Width 和中文/Latin Auto Height，作者尺寸仍保持 520×42 与 260×12；精确宽度、baseline、行距、裁剪和派生 hit rect 已由同一布局结果覆盖，不存在全局经验倍率；最新 UI 定向回归 `155 passed, 1 skipped`，Console 0 warning / 0 error。自动布局又补齐主轴 Start/Center/End/SpaceBetween，以及多个 Fill 子项受 min/max 约束后的剩余空间重分配；显式 fallback、DPI/多平台、世界透视清晰度、mask、特殊策略和 Prefab/保存/Play 仍未收口。

> 2026-09-12 再次全计划复核：当前明确勾选 `58/246`（约 23.6%）。A03.1—A03.3 的精确查询/碰撞策略、基础单轴关节及 Unity 风格固定步已经形成真实 Jolt、阶段级 Compute 批处理和 GPU→Jolt 同步证据；A04.1 又完成了公开、注册表托管的 Mesh 资源模型及运行时销毁、Play 重建、资产隔离和 GPU 在途退休主干，A02 已形成多场景编辑、脚本域、Player 事务及实际多 Camera/RenderStack 输出的纵向主链；A05 的公开 CPU `jit.compile` 也已贯通编辑、Cook、Player 嵌入和 CPU buffer/向量，B01 `inx.buffer` 已完成不可变资源描述、dense 标量/向量布局、显式传输、无复制借用与只读范围、跨帧驻留和 exact-ticket 退休，并通过真实 Vulkan kernel byte-offset、MeshRenderer 共享持有及多容量资源代际验证；B02 已新增常驻 Mesh 的资源级 Compute→Graphics 时间线依赖和独立异步 Readback，但候选性能审计、Texture 边界及完整统计未完成，因此总项不勾；A08 已固定材质、Renderer、World→View、引擎/实例、pass 与 bindless 参数域，A12 的 Tween/Sequence 运行时服务已完成真实场景验证。A01.2 的 DataAsset 已贯通 Project 创建、Inspector/Undo/保存、共享 GUID 身份、Play 缓存域隔离、原生依赖图、Player 二进制 Cook、真实 Runner Long 配置样本及版本化迁移边界，六项已全部完成。其余只有局部合同、单一平台或一次运行的实现继续保留为执行证据，不能冒充整项完成。此前“约 65%”只适用于当前正在打通的语义/物理/Mesh/Compute 核心纵向切片，不适用于整个 041。按 A00—A16、Taichi 交付及新增复合模型导入范围重新加权，当前可复用工程实现约 **60%—63%**，最终用户验收约 **42%—46%**。Windows Release 原生模块和完整 `77/77` CTest（含 16 项真实 Vulkan 测试）已经通过；最新完整 Python 回归为 `5776 passed, 13 skipped`，本轮新增定向 CPU 48 项与真实 Vulkan kernel/Mesh 2 项通过。A03.4 仍未证明动态刚体双向作用、1/8/32/128 阶梯矩阵或完整帧 300 FPS+。

本次重新勾选后的阶段判断：

| 阶段 | 当前状态 | 尚未越过的主要出口 |
|---|---|---|
| P0 | 部分成立 | A00 样本/预算未冻结；B01/B03/B05 的最终作者合同、wheel 载荷仍未完成 |
| P1 | DataAsset 子项完成，语义底座继续推进 | 完整权威目录及 MCP 同一语义仍未闭环 |
| P2 | A02 同 World 多场景7/7，精确物理/关节/固定步已有验收，Mesh有纵向切片 | 刚软体双向稳定性与数量阶梯性能、动态Mesh完整资源矩阵仍未完成 |
| P3 | Vulkan 直连、`inx.buffer`/`compute.launch` 与 compiler-only 原型成立 | 正式资源生命周期、CPU JIT 优化层、裁剪编译器发行与多平台实证未完成 |
| P4 | 已有布局/Rect/世界 UI 纵向切片 | 完整 Figma 式 UI、世界交互与透明规则、Rect Tool 全矩阵、RenderTexture/离屏 Camera 未完成 |
| P5 | 音频5/13，光标/Tween/业务基础4/5，运行时服务已有真实场景证据 | Runner Long视觉、音频剩余矩阵、业务迁移、作者工具、插件分类/Steam未完成 |
| P6 | 仅创建 041Lab | 漂亮演示矩阵、完整 Runner Long 关卡、多平台 Player、文档/CI/制品未完成 |

2026-09-09 最新部署决议：本期 JIT-only，CPU（Numba/llvmlite）与 GPU 编译工具由引擎 wheel/适用 Player 统一携带；AOT 项目构建/导出/加载暂不交付。历史 AOT 测试段落仅作记录，不能作为当前验收。双 JIT 构建收集与 Android/Web 新能力门槛见 [B06](041-buffer-jit-compute-contract.md)。源码依赖/线上仓库统一 taichi_for_infernux，位于 external/taichi_for_infernux。

状态：2026-09-10 计算设计修订并恢复实施。仅有明确证据的完整条目才勾选，局部实现、单次启动或静态性能窗口不代表所在模块完成。进度见 [执行记录](041-execution-log.md)。

目标版本：Infernux 0.4.1。顺序：040 → 041 → 042。
位置：本地 dev/，继续由 Git 忽略，不创建规划分支。后续实施、测试及交付遵循用户最新恢复指令；游戏原工程的用户改动须保留。

## 1. 完成目标和文档关系

041 用 Runner Long 的完整关卡验证引擎能否支撑正式游戏开发，不是复刻 Unity API 或运行原 C#。
交付包括通用引擎能力、inx.buffer、独立 CPU JIT/GPU Compute、裁剪后的 Taichi 计算后端、插件分类、必要的项目迁移与可分发制品。Taichi 已确定裁为引擎 wheel 内置的 Vulkan 计算编译模块，不再作为项目插件。特定玩法、XPBD 算法、描边/景深等美术效果留在项目。

- 本文是任务、依赖与验收的主计划，A00—A15 取代旧版“首批确定部分”。
- [Buffer / JIT / Compute 详细合同](041-buffer-jit-compute-contract.md)是 A04/A05 的组成部分；本轮作者 API、Taichi 改造与 V8 级 CPU 分析任务以该合同为准，旧 hpc/resident 记录只作历史证据。
- [041 权威语义详细合同](041-authoritative-semantics-foundation.md)承接原 042 的基础类型/字段/操作合同和相应任务；它是 A01 的组成部分，不是另一个可延期计划。
- [讨论决议与需求去向](041-runner-long-open-decisions.md)保留原 29 个 RL ID 和最终决定；文件名为兼容既有本地链接保留，不再是等待用户重复选型的队列。
- [042 Agentic Runtime](042-authoritative-semantics-agentic-runtime.md)消费 041 成果，只新增动态 Python 应用逻辑的发布、替换、迁移、持久化与退役等能力。

技术验证项与产品方向分开：WebGPU 计算路线、字体度量、GPU 资源互操作仍需验证；Taichi wheel 内置与纯编译职责已确认，具体编译/资源接入仍须实证。旧 Taichi CPU/Wasm 不再是默认方案。
任何必要路径未通过，不得自动删掉该平台、换成慢速 Python 循环或用另一框架兜底来宣称完成。

## 2. 基线、已存在的能力与修正

基于 2026-09-08 的源码阅读及随后讨论：
引擎基线 d0eac0f2d29c0e606288e5f2d64b5b2fd30ae203；
游戏基线 5d3fa94ef53a6569ffad2b471744732c58187849，游戏工作区有用户修改，读取时保留。
来源：E:/project/UnityProject/Runner-Long/docs/infernux-runner-long-gap-spec.md。
开工时重新记录实际基线，不将本次阅读当成执行过的 GUI/Player 测试。

| 核对结果 | 实施含义 |
|---|---|
| TemporalAAEffect 已公开导出，native 已有 camera cut/history 处理 | 补 stub、手动 reset 与多 View 验证，不从零做 TAA |
| set_inline_mesh_data 已有 native 绑定，可经 BuiltinComponent 委托调用 | 沿现有资源路径发展正式 Mesh API，不说整块写入完全不可用 |
| draw_renderers 支持 picking，要求 RG32_UINT + depth | 不重建 ID 系统；游戏的对象数据图仍需要选择集合和自定义 payload |
| 现有 Gizmo 有移动/旋转/缩放及拖拽 Undo | 复用，补全对象通用 Rect Tool 和公共 Handle 接口 |
| UIText/ImGui 字形与二维交互已存在 | 共用布局/文本/事件，增加世界绘制和射线映射 |
| AudioSource.output_bus 仅保存名称，当前回调把立体声合成 mono 再声像分配 | 音频不是只补面板；纠正声道处理、混音和实时播放基础 |
| 官方 catalog/registry 已有 category 和 targets | 增加分类浏览与规范值，不造第二个商店/注册表 |

源码入口包括 python/Infernux/components、engine/interaction、host/operations、plugins、rendergraph、renderstack、ui；
native 的 BindingScene/BindingPhysics、SceneManager、PhysicsWorld、SceneRenderGraph、InxScreenUIRenderer、InxTextLayout、AudioEngine。
实施时先用 rg 确认路径及当前实现，不能按旧目录猜测文件或整模块重写。

## 3. 共同规则

1. 本机操作在 conda activate infernux 下完成。游戏作者通过 Hub、wheel 和预编译插件使用，不手动运行 CMake/编译第三方库。
2. Fire Forced：一个正常主路径、一个权威定义；必要的参数边界、事务隔离、设备兼容和资源预算要保留，不叠加猜测式恢复、每帧 hash 或重复验证。
3. Python 命名 snake_case；serialized_field 仍为作者入口。原生字段与 Python 字段进入同一目录，更新 .pyi、Inspector、Cook、文档和 MCP 投影。
4. 保留 GUID/元数据、Content.inxpkg、资源退休队列、现有脚本事务、Undo 和固定步基础。通用机制不为游戏另开私有捷径。
5. 所有“自动”行为都有明确所有权和时间点：不隐式复制材质、不把修改 mesh 当成自动重建 collider、不用加载次序决定相机。
6. 兼容迁移只在载入/导入等边界执行，随后使用新主路径；不永久双写旧字段/新字段。
7. 多场景共享 World 不等于跨平台物理逐位确定，也不等于未来 Batch World。
8. 改动归属明确：引擎做通用资源/语义、buffer 与计算入口；Taichi 编译模块内置引擎 wheel，Steam SDK 仍由插件集成；游戏做算法/表现/数据与一次性迁移。
9. 不把代码行数作为删功能/测试的硬门槛；在迁移完成时删除被替代的 schema、UI 布局、参数缓存和重复插件逻辑。
10. 未核实的库支持、平台能力和性能写为验证工作，不写为“已支持”。

## 4. A00：固定样本和验收清单

- [x] 记录实际引擎/游戏 HEAD、工作区差异、Level02 场景和关键资产；不重置用户工程。（2026-09-08，见执行记录）
- [ ] 固定玩法流程：进入关卡、移动/拖动、卡牌/单位切换、跷跷板、顶压板、落水、重置、过关、场景切换和存档。
- [ ] 保存参考画面/交互：世界标签、红白描边、主动/遮挡斜线、水、草、景深、体积表现、中文 UI、声音。
- [ ] 记录源场景对象/顶点/实例/voice 规模、设备、分辨率与稳定帧时；后续预算根据此基线冻结，不凭空保证毫秒数。
- [ ] 游戏迁移放在独立游戏工作区；引擎只留无商业素材依赖的小型可复用 fixture。
- [x] 使用决议文档覆盖全部 29 个原始需求，并明确引擎、插件、游戏各自的验收入口。（2026-09-12：`041-runner-long-open-decisions.md` 的需求去向表经脚本复核为 29 个唯一 RL ID；每项指向 A01—A15 的引擎/官方插件/Runner Long 项目验收章节，D13 与 A05/A14 明确 Taichi 改为引擎内置、Steam 保持独立官方插件、Unity→Infernux 迁移只留游戏工程。）

验收：后续每项能指向真实消费者，不以孤立 demo 替代完整关卡。

## 5. A01：前移权威语义、DataAsset 与基础 MCP

这是其它新增作者能力的前置；详细字段与操作合同见同目录语义附录。

### A01.1 一个类型/字段定义

2026-09-08 字段目录前置：FieldMetadata 已能投影到现有不可变 FieldSchema，Python 属性事务直接复用缓存描述而非另造简化 schema，默认值由同一 codec 与初始化规范化入口处理。现有引擎 Python 组件的 137 个字段均可编译；加入 Inspector/value codec 的扩大回归最终 680 项通过。当前按需投影不等于类注册前全量编译，也不等于原生/Python 权威目录已建立；具体切换边界见 [目录迁移映射](041-semantic-catalog-migration-map.md)。

2026-09-08 局部实施：组件与 SerializableObject 已共用一次性的字段声明编译入口，移除数据对象重复解析；支持纯注解、Annotated 标记、嵌套数据/列表及显式私有字段。继承字段的运行时规范化、原始引用读出、序列化与 CDS 布局共用 get_serialized_fields；每个具体组件只分离数值存储绑定，不重复声明父类字段。声明失败不替换已注册数据类型。已验证多层继承、同布局重注册、候选布局发布/撤销，以及稳定父类下子类字段增补的真实原生读写；重载不再重复调用字段声明注册或先复制后覆盖原生槽位。扩展回归 447 项通过（14.54 秒）。权威 C++/Python 描述符、完整 owner 事务与 MCP/Player 投影尚未统一，A01 不据此勾选完成。

2026-09-08 追加修复：目标数值布局已经发布、迁移实例仍持有另一布局时，现通过同一原生 schema 事务准备独立存储代际。prepare/seal 不改变旧名称映射或旧槽位；commit 切换新代际，rollback 恢复旧映射，finalize 后旧槽位按现有 retirement 机制释放。已有目标布局持有值的提交/撤销与连续八轮字段增删均已验证。候选字段值现先走普通赋值使用的 normalize_runtime_field_value，再进入存储；改变 range 而不改变数值布局也使用隔离代际，非法范围在原生准备前拒绝。真实 CDS 覆盖范围收紧后的提交/撤销和旧 epoch 数据保留。扩展回归 455 项通过（14.02 秒），原生 CDS 合同通过（0.01 秒）。这些回归仍不代表所有字段约束、父类和子类同批更新、整条继承树迁移或完整权威目录已经完成。

- [x] 统一 native 描述符、serialized_field 编译结果与 common value codec；类注册/修订时编译，禁止 Inspector 每帧反射或从当前 JSON 值推断类型。
- [ ] 稳定 type GUID、field ID、schema version、owner、运行时可用性和迁移声明；已有资产不因路径/命名改写身份。
- [ ] 覆盖当前所有支持的字段种类、继承、列表、枚举、嵌套对象、引用、曲线/渐变、只读/隐藏/多对象编辑。
- [ ] native 注册、Python/CDS/dispatch 更新使用一致发布边界；同一类型不能出现新 schema + 旧布局。
- [ ] 统一 Inspector、保存、Cook、Python 编辑 API、基础 Command/Query、Undo/Prefab override 的字段读写规则。
- [ ] Player 在初始场景/脚本消费者之前安装 Cook 后的语义目录，保持 Editor 服务不进入 Player。

2026-09-12 字段定义主链收口：原生类型继续以 C++ semantic catalog 为作者源，Python `serialized_field` 在类注册/候选发布边界编译为同一个不可变 `FieldSchema`，两者的默认值、枚举、向量、颜色与引用均经 common `VALUE_CODECS`。Inspector 的原生 generic/multi 路径已删除按 `serialize_document()` 当前 bool/int/float/list 值猜控件类型的逻辑，改为只枚举 catalog 字段；自动化 `scene.component.schema` 同样只投影原生 catalog 或 Python 已编译字段，未声明但碰巧出现在 JSON 中的键不再成为可编辑 API。此前 Transform 的 `position/rotation/scale` 名称与本地属性错位也改由同一 schema 的 `serialized_name → field_id` 解析。Camera 的真实 MCP schema 显示 12 个声明字段、FOV 1—179 范围和两个 Projection 枚举；真实编辑器选中 Camera 后 Console 0 warning / 0 error。字段/Inspector/自动化集中回归 `81 passed`，世界 UI/Transform 路由集中回归 `40 passed`。因此本项完成；完整字段种类矩阵、跨语言发布事务和 Player 目录安装仍由后续独立条目验收。

2026-09-12 原生覆盖继续扩展到 Collider 家族：五种具体 Collider 共用 C++ 声明的 center、trigger 与 PhysicMaterial 资产语义，各自只追加 size/radius/height/direction/convex；Python 包装层不再重复维护这批类型、默认值与显示约束。PhysicMaterial 的 typed authoring reference 到既有 GUID 存储是一个显式 setter-owned 边界。真实 041Lab 用 MCP 编辑 BoxCollider 的 center 和材质并逐次 Undo 清理，Console 0 warning / 0 error；扩大自动回归 537 项、相关 C++ 测试 3/3 通过。native semantic coverage 从七种增至十二种；此增量不冒充全部字段矩阵或 Player cooked catalog，因此严格勾选数不变。

### A01.2 ScriptableObject 类数据资产

DataAsset 是暂定公共类型名，表达可独立保存的项目数据，不要求派生 Component。

2026-09-09 注册前置：修复 SerializableObject 在候选模块准备阶段提前覆盖全局类型的缺口。数据声明现在与现有 CandidateImportTransaction 共用私有准备、safe-point 提交和撤销；同批嵌套默认值使用候选类型，外部消费者仍看到已发布类型。替换模块会移除其已删除的类型，精确撤销不覆盖其它模块，跨模块抢占显式类型 ID 会在写入前拒绝。原有纯数据脚本和组件脚本的事务入口均已覆盖，扩大回归 722 项通过（15.29 秒）。这不等于 DataAsset 资产类型、GUID 身份、活实例嵌套数据迁移、完整 owner 生命周期或 Player 目录已完成。

2026-09-11 DataAsset 纵向主链：唯一 `.inxdata` 作者文档和原生 `ResourceType::DataAsset`/Importer/Loader 已接入，类型只从已发布 SerializableObject 目录解析，不按文档字符串导入模块。DataAsset 支持嵌套/集合字段、GUID 持久身份、AssetManager 共享缓存、`DataAssetRef` 与显式 `instantiate()`；Project 创建、Inspector 字段编辑、Document/Undo/自动保存均复用现有事务。原生 Importer 递归发布嵌套 GUID 依赖；Cook 将作者 JSON 确定性编码为 `Library/Artifacts/Data/<guid>.inxasset` 二进制载荷并排除原 `.inxdata`，Player 仍只按 GUID/依赖图读取。Play 使用独立 DataAsset 缓存域：运行脚本共享同一运行副本但不触碰作者对象，Stop 丢弃整域，运行副本拒绝写回作者资产；Inspector 的显式作者修改继续沿既有文档事务。数据类型现声明正整数 `__serialized_schema_version__`；无版本旧文档只在载入边界按当前字段名或 `FormerlySerializedAs` 迁移、补默认值并移除旧字段，随后只使用当前格式。Cook 总是把旧作者文档写成当前版本二进制，且不反向改写源文件；未来版本明确拒绝。Python 完整回归 `5763 passed, 13 skipped`，真实 041Lab 旧格式配置冷启动、Play 和 Stop 为 0 warning/0 error。

- [x] 用户通过 Python 类型和 serialized_field 定义数据；可从项目菜单创建资产，在 Inspector 编辑并由 GUID 引用。
- [x] 支持嵌套数据、集合、类型受限引用及关卡规则所需的已注册多态数据；类型判别来自权威目录，不能按任意字符串 import 类。
- [x] 同一资产被多个 Scene/Prefab 引用时共享身份；需要可变副本时显式实例化，不隐式修改所有引用者。
- [x] Play 编辑隔离与显式保存沿现有事务；Player 中配置资产内容不可写回 Content.inxpkg，运行状态和存档另有归属。
- [x] 类型重命名/字段迁移沿统一版本合同；资产引用、默认值、Undo、导出重开均可往返。（2026-09-11：稳定 type ID 与显式 schema version 解耦 Python 类名；旧文档在唯一载入/Cook 边界按声明式 former name 迁移并补当前默认值，当前版本保持严格字段集合，未来版本拒绝。真实旧格式 RunnerLongConfig 冷启动和 Play 通过，完整 Python 5763/5763。）
- [x] 用单位规则、章节/关卡配置、视觉参数和音频分组配置作真实样本。（2026-09-11：`Infernux041Lab/08_DataAssetConfiguration` 通过一个 GUID 资产保存四组嵌套配置；MCP 沿同一 Project/Document 事务创建和修改，Python 组件在 Play 中读取并修改隔离副本，Stop 后作者值保持 7.25；最近一次真实运行 0 warning/0 error。）

### A01.3 基础操作与 MCP

2026-09-12 MCP 大型动态 Mesh 投影：`infernux.scene.hierarchy.get` 默认不再把运行期 `inlineVertices/inlineIndices` 整体复制进场景概览，而是稳定返回 `inlineMeshSummary.vertexCount/indexCount`；确需逐元素检查时必须显式传 `include_mesh_data=true`，不是按长度截断或失败后重试。真实 `02_InteractiveSnow` Play 中，65,536 顶点/390,150 索引的默认响应由 62,692,852 字符降至 15,763 字符；显式完整查询仍返回原数组。两份 MCP Python 源在编辑器运行期间更新后由现有插件 preload/资源刷新主路径生效，无需重启，随后查询与 Game Play 均为 0 warning/0 error。协议仍在插件内手写通用场景投影，尚未完成从引擎权威目录枚举生成，因此下面总项不提前勾选。

- [x] 场景、对象、组件、字段、资产查询和编辑操作归引擎，MCP 仅做协议投影；没有 MCP 插件时也能调用。（2026-09-12：31 个 Scene/Asset/DataAsset 作者操作、schema 构造、主线程调度和 GUID 身份解析迁入 `Infernux.host`，编辑器在插件加载前注册；MCP 生命周期只移除自身 67 个操作，关闭后引擎操作仍在。独立无插件 registry 与完整 MCP 98 项目录回归通过。）
- [x] 基础操作有明确输入/输出、错误、线程/阶段、权限、Undo、Editor/Player 可用性；不全部声明返回 object。（2026-09-12：31 项核心操作具有各自顶层输出字段/必需键、返回验证、稳定错误、owner thread、editor.read/editor.authoring 阶段、capability、reversible 与 availability；真实 Light 与 DataAsset schema/结果符合合同。）
- [x] MCP 枚举引擎目录并刷新 schema；删除插件内重复字段和通用操作定义。（2026-09-12：网关动态读取同一 registry revision；插件内 Scene/Asset/DataAsset 文件仅为兼容 re-export，定义只存在于 `Infernux.host`。真实旧 MCP 0.1.1 仍可投影新 98 项目录，但不能覆盖引擎 owner。）
- [x] 用 native 组件、Python 组件、DataAsset 验证 Inspector/Python/MCP 同值同约束，一次编辑一次 Undo。（2026-09-12：集成回归同时覆盖 native Light 0—10、Python serialized speed 0—20 与 DataAsset 嵌套字段，三者沿统一 schema gateway 拒绝未声明字段并逐编辑产生逐 Undo；真实 041Lab 的 RunnerLong DataAsset 经 MCP 从 7.25 改为 6.75 后，一次编辑器 Ctrl+Z 立即恢复 7.25，实时文档、持久化事务和 Inspector 控制器共享同一权威对象。聚焦回归 68 passed，真实 Console 0 warning/0 error。）
- [ ] 新生成脚本的 prepare/publish/retire 操作不在此实现，留给 042；这里只完成 authored/plugin 生命周期所需注册事务。

验收：附录 S0—S6 完成；所有基础消费者共享真实定义，不能只把数据复制到一个新注册表而旧消费者继续各行其是。

## 6. A02：同 World 多场景

- [x] Additive 加载相当于引入另一 Scene 的对象：脚本、物理、查询、渲染、声音统一参与当前 World。
- [x] 保留所属 Scene 以便保存、卸载、移动归属；active scene 决定默认新对象归属，不使其它驻留 Scene 暂停。（2026-09-11：公开根对象跨 Scene 移动保持对象/组件身份；Hierarchy 双驻留场景回归证明新对象进入 active Scene，原 Scene 继续驻留并更新。）
- [x] 沿现有 prepare/activate 增加驻留、激活、卸载、取消；区分构建场景列表与当前已加载场景列表。（2026-09-11：Editor 与 Player 共用事务；真实 PlayerRuntimeAssetCatalog Cook 场景完成 Additive prepare/publish，取消/失败隔离和 loaded/build 目录已有集中回归。）
- [x] 多 Camera/RenderStack、灯光/环境、AudioListener 使用与单场景相同的规则；核对并明确现有规则，不按加载先后覆盖。
- [x] 场景间对象/组件引用、Find/遍历、碰撞、脚本阶段和编辑器选择都覆盖已加载内容。（2026-09-12：公开查找/引用、跨 Scene Jolt 碰撞、native 生命周期、脚本热重载/删除和 Hierarchy 选择统一覆盖 resident Scene。）
- [x] 卸载 B 不销毁 A 的对象；明确常驻对象与依赖引用的卸载行为，不静默替换相似对象。
- [x] Play clone、多场景编辑保存和 Undo 保留各自归属；新 ID 和引用不碰撞。（2026-09-12：全部 resident Scene 一起进入/退出 Play 脚本域并恢复各自文档 revision；双场景保存/Undo 与原生/Python 身份重映射已有集中和真实编辑器回归。）

2026-09-11 同 World 多场景底座：原生生命周期、固定步累积、Transform 帧缓存、物理 World、Renderer/Light 注册表已从 active Scene 边界扩展到全部 loaded Scene；active Scene 只保留默认作者归属和主视图策略。公开 `LoadSceneMode.SINGLE/ADDITIVE`、loaded Scene 目录及 Player prepare/publish 事务已接入；复制同一场景文档时，GameObject、原生/Python Component ID 在原生提交阶段统一重映射，提交令牌把同一映射交给 Python `GameObjectRef/ComponentRef`，不在引用解析时猜测。多 Camera 继续按 `Camera.depth + component_id` 稳定排序，环境仍由 active Scene 决定。原生回归覆盖双场景生命周期、固定步、Transform、灯光、Camera 排序、32 轮加载卸载、卸载隔离和身份重映射；Python 真实回归覆盖公开 Additive 文件加载、跨语言引用、卸载后引用失效而不误绑相似对象、World AudioListener 和 A 球落到 B 地板。常驻对象继续沿已有 persistent Scene 合同，因此卸载语义项完成。RenderStack 实际出图、编辑器多场景保存/Undo/选择及完整 Player 实机加载仍未完成，其它项目不提前勾选。

2026-09-11 编辑器纵向切片：Hierarchy 以一个虚拟滚动列表展示所有 resident Scene，场景标题与对象拥有稳定语义 ID；标题点击经统一编辑命令同时切换 native active Scene 与其 `SceneFileManager` 文档，不再只换 World 指针。每个加载场景现在绑定独立的资源路径和 `DocumentRegistry` 身份，场景 B 的属性、结构 Undo、dirty revision 和保存目标不会因随后激活 A 而漂移。真实可见 Infernux041Lab 通过 MCP 加载 `06_SharedMeshPublication`，搜索并点击第二个场景标题后 active world 从 1 切到 3，临时重命名与 Ctrl+Z 都作用于第二场景且 Console 无错误；集中回归为 185 个场景测试、69 个编辑器命令/MCP 测试。Single 替换时多个脏场景的统一关闭事务、Play clone 与完整保存/卸载矩阵尚未完成，因此 A02 组合项仍不提前勾选。

2026-09-12 Play/脚本域收口：`PlayModeManager` 在同一个 Play 边界捕获全部 resident Scene 及各自 `DocumentRegistry` revision，逐 Scene 建立新的 Play Python 实例，Stop 后恢复所有作者图和进入前的 active Scene/文档；运行时改名和字段修改不会写回任一作者场景。Play 中的 Single 加载即使销毁 Additive native Scene，Stop 也会先删除运行期独有 World、按快照重建缺失 Scene，再原子替换 Scene-to-document 绑定；不保留悬空 native 引用。编辑态 Open/New 的 Single 替换也以一个既有关闭事务顺序处理全部 resident Scene 文档的脏状态，提交后卸载其它 Scene。场景文档事务清空进程级 Python 实例索引后会重挂全部 resident/persistent Scene，不再使一次 Additive 发布把原场景移出生命周期表；脚本 reload、删除和 MissingScript 恢复也从 active-only 改为整个 resident World。MCP 的 Stop 完成条件同时等待 Edit 状态和场景恢复任务结束，不再在状态先切换时过早返回；相关操作仍只依赖统一 Host API。集中回归 `295 passed`；真实可见 Infernux041Lab 双场景 Play/Stop 为 0 warning/0 error，最近一次进入约 669.3 ms（其中 native Start 约 648.6 ms），退出恢复约 75.7 ms，并由同一次 MCP 响应返回 `transition_pending=false` 和 exit timing。RenderStack 实际出图仍是后续边界，不据此勾选 A02 总项或 Camera 项。

2026-09-12 多 Camera/RenderStack 实际出图：MCP 增加经统一 Host API 切换 resident active Scene 的操作，只用于暴露现有权威场景能力。真实可见 Infernux041Lab 同时加载 `01_XPBD_Jelly` 与 `06_SharedMeshPublication`，在第二场景临时创建独立 Camera 和 RenderStack；Camera 继续按 `depth + component_id` 排序，并以 `DepthOnly` 接续第一相机。1920×1080 GPU 捕获显示第二相机开启时从另一视角重绘同一 World 的两场景几何，关闭它后立即恢复第一相机视角，证明不是 active Scene 或加载先后覆盖。临时对象随后删除并通过 discard reload 清除作者 dirty 状态，项目恢复为单一原始场景。结合既有跨场景 Camera 稳定排序、active Scene 环境策略、Light 注册表和 World 唯一 AudioListener 回归，本组合项完成；RenderTexture/离屏 Camera 仍属于 A10，不混入本项。

验收：A 的球落到 B 的地板上；双场景与等价单场景的相机/交互结果一致；切换 active scene 不暂停另一个；反复加载卸载无泄漏。

## 7. A03：精确物理、关节与固定步扩展

### A03.1 几何与碰撞

2026-09-09/10 查询局部：Physics.compute_penetration 提供预测世界 origin/rotation 下的分离方向、深度和两表面接触点；Box/Sphere/Capsule/Cylinder 与已经发布的凸 MeshCollider 使用与物理相同的只读几何构建（含 center/缩放/轴向），不写 Transform、不同步 broadphase、不应用 layer/trigger 过滤。指定单个 compound 成员，禁用的基本 Collider 也可查询。真实 Jolt 回归覆盖预测分离、旋转/轴向、负缩放/center、世界不被移动、凸 Mesh 穿透以及不支持 shape 的显式错误。非凸或尚未完成 cook 的 MeshCollider 仍明确拒绝，不能偷偷启动异步任务；批量查询热路径仍未收口。

- [x] 补 Collider.raycast、closest_point、Physics.compute_penetration。世界输入/输出，预测姿态查询不改 Transform、刚体或 broadphase。
- [x] 穿透输出为把 A 推离 B 的单位方向与距离；无交叠返回 None；不支持的 shape 明确报错。
- [x] 覆盖 Box/Sphere/Capsule/Cylinder/已 ready 的凸 Mesh 及实际 center、旋转、支持的缩放/compound；不承诺任意非凸对的全局最小分离解。
- [x] 直接指定 Collider 的纯几何查询不暗中使用 layer/trigger/pair ignore；世界 cast/overlap 保留明确过滤。
- [x] 组件级 ignore_collision/get_ignore_collision 正确作用于 compound 子对象，切换事件在指定物理边界生效，销毁清理归属。
- [x] 为热路径批量查询提供复用输出接口，不默默截断；测量 Python 对象分配，不能把复用 list 冒充零分配。
- [x] 暴露项目实际需要的接触点、法线和求解后冲量；注明阶段、方向和单位，不用质量乘速度伪造 solver impulse。

2026-09-10 通用接触数据边界：`Physics.set_contact_event_stream_enabled()` 现可按需开启最近固定步的解析接触流，`Physics.get_contact_events()` 以 NumPy SoA 返回事件类型、body/sub-shape 身份、世界接触点、法线和相对速度；默认关闭，不给普通项目增加事件缓冲成本。新增 `Physics.set_contact_impulse_stream_enabled()` / `get_contact_impulses()`，在 Jolt 临时约束释放前快照实际速度求解器 lambda，返回世界接触点、A→B 法线方向以及求解后线冲量；同样默认关闭，且不把质量乘速度伪造成 solver impulse。该接口明确位于 Jolt Step 完成后的 resolved snapshot 阶段，反馈仍由耦合器通过现有批量 `apply_rigidbody_impulses` 提交。因此接触点/法线/真实求解冲量的公共边界已完成并有真实下落球回归；GPU 驻留接触输出、多接触归并和 300 FPS 双向门槛仍未完成。

2026-09-10 单 Collider 查询纵向切片：新增 `Collider.raycast(origin, direction, max_distance)`，直接锁定该 Collider 已发布的 Jolt body，并按 compound subshape 身份跳过同 GameObject 的兄弟 Collider。它使用当前世界姿态和 Jolt 表面法线，不应用 layer、trigger 或 pair-ignore 过滤；真实测试覆盖 IgnoreRaycast 层上的 trigger Box、前方兄弟 Sphere、旋转 Box 和非法射线。`Collider.closest_point(point)` 使用同一 Jolt 几何：内部点原样返回，外部点返回目标形状表面最近点，覆盖 Box/Sphere/Capsule/Cylinder、ready 凸 Mesh 与 compound 兄弟隔离。世界 `Physics.raycast` 的过滤合同保持不变。批量复用输出仍未完成。

2026-09-10 精确碰撞策略收口：新增公开 `Physics.ignore_collision/get_ignore_collision`，策略使用完整 64 位 Component 身份并在 Jolt `OnContactValidate` 按实际 compound subshape 拒绝单个接触，而非扩大为整个 body pair。运行时切换会清理事件 pair、使接触缓存失效并唤醒相关 body，因此下一物理边界即使用新策略；Collider 销毁时移除归属。真实回归证明同一 compound 的 Box 保持碰撞而指定 Sphere 被忽略、解除后恢复接触，并验证重复设置是幂等状态写入。

2026-09-10 批量查询收口：新增 `Physics.raycast_batch(origins, directions, out, ...)`。输入固定为连续 `float32 (N, 3)`；命中、点、法线、距离、body/component/object 身份写入调用方提供的 typed NumPy SoA，容量可大于 N 且只改写有效前缀，容量不足在写入前报错。世界 Transform/broadphase 每批只同步一次；默认含 Trigger 的最近命中使用 Jolt closest collector，不再先构造完整 hit vector。256 个真实命中的 Python 跟踪峰值为 55 bytes，结果字典和全部数组身份保持不变，未创建逐命中的 Python 对象。

- [x] 批量查询的原生 Jolt 阶段释放 Python GIL；只在写回调用方数组时重新取得解释器锁。该边界不改变查询快照、命中顺序或输出格式，并由 Release wheel 与 29 个物理查询回归验证。

### A03.2 关节

- [x] 基础 Hinge 复用 Jolt：连接刚体/世界、锚点、轴、角度限位和是否互撞；跷跷板不需要 motor/spring 才能验收。
- [x] 提供顶压板所需单轴平移限位，锁住其余自由度，游戏自行施力；使用 Jolt 对应约束，不逐帧强制 Transform 假装关节。
- [x] 公共封装可采用单轴组件或六自由度约束的明确子集，由实现核对选最小一致模型；完整驱动/projection/break 功能不是隐含任务。（采用 HingeJoint + SliderJoint 两个职责明确的单轴组件。）
- [x] 关节字段、连接引用、Inspector、Prefab、复制、Play clone、Disable/Destroy/Stop 生命周期贯通。
- [x] 关节禁止互撞与游戏设置 pair ignore 分别持有归属，解除关节不清除游戏的设置。

2026-09-10 基础 Hinge 收口：新增原生 `HingeJoint`，直接持有同一 `PhysicsWorld` 中的 Jolt Hinge 约束，不用逐帧覆写 Transform。真实物理测试覆盖连接固定世界后的锚点保持、Z 轴单轴摆动、±35° 限位、连接 kinematic Rigidbody，以及运行时切换连接体互撞后重新产生接触。约束创建延迟到 Collider body 发布后的同一固定步，Disable/Destroy 或 body 重建时释放；连接引用使用稳定 component ID，并通过通用 native reference remap 在 GameObject clone 和另一 live Scene 的文档复制中指向新图。后续作者链证据见下一段。

2026-09-10 关节作者链收口：新增原生 `SliderJoint`，直接使用 Jolt SliderConstraint 提供世界/连接刚体锚点、局部轴、正负距离限位与连接体碰撞开关；约束锁定其它平移轴和全部旋转，自由度模型不扩张为当前没有消费者的通用六自由度、motor、projection 或 break 系统。Hinge/Slider 的 `connected_body` 统一由原生语义目录声明为 COMPONENT，但文档落盘字段明确映射到 `connected_body_component_id`；内置组件 Inspector 也获得可复用的原生组件引用选择器，编辑、Undo/Redo、Prefab/文档实例化、层级复制、live Scene 复制、Disable、Destroy 与 Stop 重建均有回归。实际 041Lab 编辑器通过 MCP 创建两种关节并读取相同权威字段后删除临时对象。关节持有引用计数的 body-pair 抑制，游戏 API 持有幂等的精确 Collider-pair 策略；真实回归证明解除关节抑制不会清除游戏策略，解除游戏策略后接触才恢复。

### A03.3 Unity 风格固定步

- [x] 沿现有 RunFixedSimulationStep 和 fixed_update 实现时间累积，一帧可执行零次/多次固定步。
- [x] 明确 fixed_delta_time、time_scale、暂停/单步与最大追赶时间，防止补算无限积压；不把超时当成程序崩溃。
- [x] 公开物理输入提交前、求解完成并同步结果后的受控扩展阶段；准确记录接触事件和 Transform 回写的位置。
- [x] 手动步进使用同一 World 物理路径，与自动模式互斥；明确是否执行脚本阶段，不同操作不能都叫 simulate 却各自推进时间。（2026-09-10：真实 Jolt 集成回归验证未暂停 `step()` 不推进；暂停后恰好执行一个固定步，并按 `fixed_update → physics_pre_step → physics_post_step → update → late_update` 顺序执行。）
- [x] Taichi/项目任务批量接入，声明读写及完成依赖，不能在回调里私开另一个物理循环。
- [x] CPU 刚体与 GPU 求解需要同一步双向数据时显式同步并测量成本；允许延迟的数据由项目明确采用，不能伪装同一步结果。
- [x] 固定步、渲染插值和相机显示更新各有职责；不承诺不同硬件/后端浮点逐位一致。

2026-09-10 固定步首段收口：直接扩展 `SceneManager::Update/RunFixedSimulationStep/Step` 的生产路径回归，证明不足一个 fixed delta 的帧执行零次、累积到阈值执行一次、长帧按 `maximum_delta_time` 截断后可执行多次；`time_scale=0` 不推进 fixed，`time_scale=2` 以双倍真实频率推进但每步游戏时间仍为一个 `fixed_delta_time`，对应 unscaled fixed time 为除以 scale。Pause 后普通帧不推进，显式 Step 恰好推进一次同一物理路径。没有新增第二计时器、超时重放或异常恢复。扩展阶段、GPU/CPU 同步和完整插值职责仍按后续未完成项推进。

2026-09-10 扩展阶段收口：在同一个 native fixed frame 和同一份 immutable lifecycle snapshot 中公开 `physics_pre_step` 与 `physics_post_step`。生产顺序固定为 `fixed_update → Collider/Transform 同步 → physics_pre_step → Jolt → 接触事件 → Rigidbody/Transform 发布 → physics_post_step`；没有第二计时器、第二物理循环或项目侧调度器。真实 Jolt 测试证明 pre 阶段施加的 VelocityChange 进入同一次求解，post 阶段立即读到本步权威速度与 Transform。CPU/GPU 同步成本和任务依赖仍由后续条目验收。

2026-09-10 固定步 Compute/显示职责收口：共享生命周期调度器现在以整个 phase 为唯一录制边界，Editor 与 Player 不再因组件数量把同阶段 GPU 工作拆成多次提交。裁剪编译器已有的外部 buffer READ/WRITE 分析被保留到 Infernux task metadata，并传入 RHI；同一提交按实际 RAW/WAR/WAW 冲突插入 Vulkan 屏障，完成和安全退休继续使用既有 submission ticket，不增加第二套任务图。真实 Vulkan→Jolt 回归将编译后的 kernel、两份 `vector3` 反馈合入一次提交/等待，2 个刚体只回读 48 字节，当前机器稳态等待约 0.47 ms；紧随其后的同一 Jolt 步按该速度更新位置。允许延迟的视觉数据保持 GPU 驻留；调用 `get_data` 或 CPU 物理消费才是明确观察边界。显示侧继续由固定步写入 previous/current 物理姿态，按 accumulator alpha 生成 presentation Transform，随后 Update/LateUpdate 的相机逻辑和最终渲染提取消费显示姿态；不承诺跨硬件浮点逐位一致。

验收：旋转预测形状、compound 屏蔽、跷跷板/顶压板、不同显示帧率下固定步次数、暂停单步与任务阶段都有行为测试；XPBD 算法仍归游戏。

### A03.4 Sphere 软体与普通刚体的交互（2026-09-09 修订验收）

**当前验收场景（替代此前自建网格果冻，不是新增一套并行验收）：** 桌面 Infernux041Lab / 01_XPBD_Jelly 使用引擎内置 Sphere 的原始顶点、UV 和索引，对该 Sphere 做真实软体变形；允许内部体积计算网格，但不能另外生成可见球面取代引擎 Sphere。球体与地面上的多个真实 Cube / BoxCollider 交互，使用 Assets/Materials 下持久化 .mat 文件及序列化 GUID 引用，不由脚本临时创建外观材质。

- [ ] Taichi 性能硬门槛为 **300 FPS+**，即真实编辑器 Game Play 完整帧平均小于 3.33 ms；不是旧场景数据、仅 GPU kernel 耗时、扣掉编辑器后的 game_only 或静止数字。固定当前 Sphere 拓扑、343 节点、1296 四面体、9 子步、50 Hz 物理、1920×1080 及文件材质，覆盖落下碰撞、重新下落、持续推挤多个 Cube。报告完整帧和游戏/GPU分项，以及 P50/P95/最大值；首次编译单列，禁止降低质量或关闭交互达标。
- [ ] 功能验收：Sphere 原始拓扑与 UV 不变；文件材质可在 Inspector 修改并持久化；方块位置/旋转/尺寸由场景 Collider 读取；接触下发生形变并恢复、无明显穿透/飞散；按键重置、推动、内部节点 Gizmo 可用。
- 当前静态 Cube 投影只覆盖软体对障碍物的响应，不替代下述动态刚体双向冲量/摩擦/质量/多接触验收；旧果冻短窗口的 400 FPS 数字只保留在执行历史，不沿用为本场景通过证据。

**最终门槛再次明确（用户追加）：** 300 FPS+ 必须发生在真正的 Rigidbody 与自定义软体双向交互时，并且不能因增加刚体数量而大幅下降。静态障碍物演示、单接触公式测试和扣除耦合成本的性能数字均不能替代此门槛。

**引擎改动边界（用户最终强调）：所有引擎侧更新服务于整体引擎，Sphere 场景只提供验收负载，不提供特例。** 不引入按场景名、GPUJelly 类型、球体尺寸、固定刚体数量触发的快速路径；不把 XPBD/Neo-Hookean 常量、Sphere 蒙皮映射或特定方块接触代码放进 core。项目算法与插件编译实现、引擎通用资源机制分层维护。

- 通用访存：复用已有计算/渲染资源所有权和队列，完善跨帧 GPU 驻留、明确的 CPU/GPU 读写边界、按需/按范围批量传输、staging 复用与内存可见性。统一内存设备与独显以真实设备能力决定存储策略，不承诺一律零拷贝；不靠逐帧全量哈希猜测 NumPy 是否修改。
- 通用同步：以生产者/消费者任务依赖决定提交与等待；异步回读、完成通知和安全销毁属于引擎机制，不由某个场景隐藏 fence 或绕过安全边界。旧 compute.batch 在迁移前保持已有 NumPy 完成合同；新 buffer+launch 明确驻留，CPU 通过 get_data 取结果，不偷偷改变旧调用行为。
- 已有基础不能重造：插件 backend.py 已有私有 _ResidentState 和引擎队列提交。按 A05 将可复用存储/提交机制迁入引擎 buffer 主路径，替换旧 batch/resident 作者 API，不并列创建第二套场景专用管理器。

2026-09-09 通用访存局部实现：复用 _ResidentState，copy_to_host(*arrays) 支持选择性批量回读，新增 copy_from_host(*arrays) 显式发布选定 NumPy 输入；无参数仍传全部数组。按已接纳存储/布局匹配并合并完全别名，拒绝未声明子视图/形状变更/不可写回读，整批检查后再进入一次原生传输。不检查数据内容、不推测 dirty、不自动替换未选中设备状态。23 项 preload/生命周期单测、完整真实 .inxpkg GPU/JIT/AOT 包回归 7 项通过；混合大型计算状态、小控制输入和小反馈用例验证未选中 NumPy 镜像可保持过时，显式上传不会覆盖其它设备数据。本入口目前仍是内部后端机制，不是公开 resident API、异步回读、范围传输或渲染直连已完成。随后已将原生 _transfer_numpy 的前置 synchronize 改为 flush 后提交传输，依赖既有同队列入口内存屏障保证顺序，保留同步传输完成及 Program 清理；空传输不触发提交。Device::upload_data/readback_data 仍逐数组创建 staging 并 submit_synced，通用 staging 复用、异步边界与性能收益仍需继续实现及测量。
- 通用消费：GPU 计算生成的数据能由既有 Mesh 渲染或下一段计算直接消费；法线/切线/bounds 工具依资源所在位置执行，CPU 手动编辑路径仍有效。Mesh 不因本演示而暗中携带软体 Collider 或触发每帧 recook。
- 通用物理接口：提供权威刚体/形状的批量快照、生命周期和批量冲量提交；Jolt 的整体世界和固定步规则不被替换。具体 XPBD、接触求解和材质行为在项目/插件实现。
- 可复用性验证：每项 core 改动必须有脱离该场景的契约测试，并以非软体的普通数组计算、通用动态 Mesh 等第二消费者验证；验收包括 CPU 读取、GPU 连续消费、异常/销毁/卸载时资源生命周期。没有第二消费者的证据，不把场景脚本中的加速代码包装成“引擎优化”。遵守 fire-forced，不扩张不必要的兜底、自动修复或重复状态管理。

- [ ] 扩展性测试先采用 1 / 8 / 32 / 128 个实际 Jolt Rigidbody 的阶梯矩阵（测试规模，不是引擎数量上限）。分别测远处无关刚体、附近运动但未接触刚体、集中多点接触刚体；分开报告场景总刚体数、有效候选对数和活跃接触数，不混为一个指标。每档至少覆盖持续交互、重置与重复 Play；固定硬件、画质、软体规模、固定步及物理子步。
- [ ] 记录完整帧平均/P50/P95/P99/最大值、Jolt 步进、候选生成、GPU 窄相/接触求解、Mesh 变形与法线、上传/回读字节数、提交次数和 CPU 等待时间。给出随总刚体/实际接触数量增长的曲线；不能将真实接触工作量增长承诺为无限常数成本，必须证明远处无关对象不扩大粒子×刚体全遍历，并保持测试矩阵内的 300 FPS+。

2026-09-10 引擎侧批量路径推进：新增 `Physics.query_rigidbody_box_states_in_bounds`，在一次原生调用内完成 Jolt broad-phase AABB 候选、层/Trigger 过滤、世界空间 BoxCollider 描述和可复用 GPU SoA 缓冲提交；Python 只接收候选包装器和有效前缀。该接口用于通用 GPU 软体/接触求解，不是 041Lab 特判。现有 `get_rigidbody_states`、`get_rigidbody_box_states`、冲量写回仍属于 API 底座，几何窄相、GPU 驻留接触反馈、固定步耦合和 1/8/32/128 刚体矩阵仍未完成，因此 A03.4 继续保持未勾选。
- [ ] 引擎层建立可复用的耦合热路径：复用 Jolt World 的形状身份/生命周期与空间索引，在软体包围体附近筛选候选；按最终性能证据决定宽相所在 CPU/GPU，但热路径禁止 Python 逐粒子调用 compute_penetration 或全场逐对查询。GPU 窄相与多接触迭代不遍历所有远处刚体。
- [ ] 软体位置、速度、约束和变形网格在固定步之间驻留引擎共享 Vulkan 设备，复用现有 RHI/队列/同步；不另建 Vulkan core。不为显示每帧回读整个软体再重新上传 Mesh。外部数据入口为 inx.buffer/标量，NumPy 可选；GPU 存储及任务依赖由引擎管理。
- [ ] Jolt 继续负责刚体世界，不以“全物理搬到 GPU”作为未经证明的前提。固定步批量上传相关刚体权威姿态/速度/惯性与形状更新，GPU 按刚体归并反馈，只回传紧凑线冲量/角冲量；明确读写、等待及实际 Jolt 步进的顺序和代际。支持动态、静态、运动学体，保证零反馈不唤醒休眠体，禁用/销毁后无悬空引用。
- [ ] 同一演示验证刚体压入球体、球体推动刚体、偏心接触转动、多个刚体同时挤压、摩擦与质量差。动量/能量/穿透/体积与长期稳定性和帧率同时达标，不能通过降低接触质量、改变时间尺度或让物体尽早睡眠来换取分数。

2026-09-09 局部接入：Rigidbody 新增 get_point_velocity 和 caller-owned NumPy float32 (N,3) 批量 get_point_velocities，读取真实刚体速度/角速度/质心，不使用显示 Transform。偏心 Impulse 对实际 Jolt 球体的线速度、角速度、接触点速度与不重复消费已测试；已发布 body 的 Impulse 立即生效，因此耦合器仍须明确在固定步哪个阶段提交。这不是完整软体接触或双向耦合完成。

2026-09-09 批量耦合接口：Physics.get_rigidbody_states(rigidbodies) 返回按输入顺序的独立 float32 NumPy 数组（position、rotation xyzw、center_of_mass、linear_velocity、angular_velocity、世界轴 inverse_mass 对角项和世界空间 inverse_inertia）。每个 body 在同一读锁内从 Jolt 取值，冻结轴直接遵守 Jolt 掩码，运动学/静态逆质量和逆惯性为零；没有实际 Collider body 时明确报错。Physics.apply_rigidbody_impulses 接收两个 (N,3) 数组，分别为世界线冲量、关于各自 COM 的角冲量，沿既有 Impulse 路径立即提交；不乘 dt、零值不唤醒刚体、不自动推进物理。输入检查在任何反馈写入之前完成。

2026-09-09 空间候选局部接入：`Physics.query_rigidbodies_in_bounds(minimum, maximum, layer_mask, query_triggers)` 直接使用 Jolt BroadPhaseQuery 的 AABox 查询，返回范围内已启用且实际拥有 Rigidbody 的唯一候选；静态纯 Collider 和远处刚体不进入结果。真实测试在 3 个近场刚体、128 个远场刚体和近场静态 Collider 下只返回 3 个候选。它消除了软体耦合器按世界总刚体数做 Python 全量扫描的必要性，但仍只是宽相候选，不代表任意形状窄相、多接触归并或 GPU 反馈链完成。

同日补充 Box SoA：`Physics.get_rigidbody_box_states` 一次输出候选刚体上启用的 BoxCollider，包含输入 body 索引、与 Jolt 构造一致的世界中心/旋转/半尺寸，以及摩擦和弹性；`out` 可使用容量更大的 CPU `inx.buffer`，以 `count` 标记有效前缀。多 Collider 会扁平化，Sphere 等不会塞进含糊的 union 参数。旋转、非均匀缩放、center、混入非 Box 刚体及 buffer 复用测试通过。它完成当前 Cube 接触所需的权威输入切片，但 GPU 窄相、紧凑反馈回读及固定步接线仍未完成。

真实 .inxpkg 的 GPU 接触回归使用上述入口，将给定的偏心接触分别施加到普通球体与旋转非均匀箱体，验证双方线动量、角动量和接触法向速度。该测试只覆盖每刚体一个给定接触的无摩擦冲量求解，尚无几何接触检测、完整果冻、多接触归并/迭代、GPU 驻留或场景性能验收；不据此勾选 A03.4。

后续局部更新：上述 GPU 回归已改为通过 compute_penetration 检出小球与实际球/箱的几何接触，再传入 GPU；已不再手工指定接触点与法线。无摩擦球面法向经过球心，因此要求球体不凭空自转，而偏心箱面接触应产生转矩。仍只有每刚体一个接触；CPU 几何查询不是最终 GPU 驻留碰撞热路径，多接触求解和果冻场景验收仍未完成。

- [ ] 通过权威 Collider/Rigidbody 状态参与同一 World，不再只碰撞脚本写死的地面。真实样本包含动态球下落、偏心刚体撞击、Box 压块、静态地面和 kinematic 推板；解析球测试不代表任意 Collider 支持，形状边界与 A03.1 一致。
- [ ] 刚体推动软体变形，软体接触求解产生的反作用冲量与力矩反馈到实际 Jolt Rigidbody。不是只让果冻躲开球，也不是改 Transform 伪造受力。覆盖摩擦、质量差、静态/运动学物体、多个接触与偏心转动。

- [ ] 建立通用的可变形对象 Transform Anchor，而不是让 041Lab 私有脚本修补 Gizmo。锚点以一个紧凑的 GPU pose 表达世界 position、quaternion rotation 与 local scale；模拟到 Transform 只做一次异步控制面回读，W/E/R、Inspector 与脚本对 Transform 的修改则在下一物理边界获得一次写权限，并以 `new TRS * inverse(old TRS)` 直接作用到登记的驻留点/向量状态，随后把权威交还模拟。禁止双向每帧互抄造成反馈环，也禁止为此回读整份软体。

- [ ] Anchor 的双向规则覆盖 position、rotation、scale：模拟可发布整体位移和旋转；scale 是明确的作者形状修改，必须同步重建软体 rest tensor/嵌入偏移等派生数据，不能从每帧挤压形变反推 Transform scale。创建运行时对象和进入 Play 时，必须先用已作者化的 Transform 初始化模拟，不得用默认 pose 反向覆盖场景中的 TRS。验证平移、旋转、非均匀缩放后渲染网格、Scene 选择描边、W/E/R 工具、内部核 Gizmo 与碰撞状态处于同一空间；Play/Stop、场景卸载和脚本热重载后绑定不得保留失效原生 Transform 指针。

2026-09-11 Anchor 启动交接修正：`bind_transform` 现在可以显式接收模拟初始 pose，并在返回绑定前由场景中已作者化的 Transform 获得权威，把与初始 pose 的 TRS 差一次性应用到登记的 GPU 点/向量状态及派生 rest 数据，使首个 FixedUpdate 也不会观察到默认尺寸，随后才交还模拟发布。真实可见 `Infernux041Lab` 验证中，编辑态 TRS `(-0.75, 2.8, 0.5) / (15°, 35°, 8°) / (1.35, 0.70, 1.15)` 进入 Play 后 scale/rotation 保持，位置继续由模拟更新；Stop 后精确恢复进入 Play 前的完整 TRS。集中 Mesh/compute 回归 `54 passed`，当前无运行时错误。Anchor 的卸载/热重载完整矩阵仍与本条一同保留未勾选。

2026-09-11 世界空间驻留 Mesh 的编辑器空间修正：渲染顶点继续以 identity matrix 直供 GPU，但 MeshRenderer 在绑定时记录 Transform anchor，后续用 `current world * inverse(binding world)` 单独推进保守 bounds。RenderWorld 分离 render matrix 与 bounds matrix，因此 Transform 更新会统一驱动相机裁剪、Scene 拾取、选择描边和 Rect Tool，不回读 GPU 顶点，也不在果冻脚本里写特例。真实 041Lab Play 中 Rect 控制柄已与移动后的软体重合；Vulkan 回归覆盖绑定后的 position/scale 变化。本修正只收掉显示/编辑工具滞留，完整双向 Anchor 生命周期仍保持未勾选。
- [ ] 复用固定步和已有力/冲量入口，核对 add_force_at_position/Impulse 的消费阶段。补齐热路径所需权威姿态、质心、接触点速度、惯性数据的批量读取和按刚体归并反馈的批量提交；不能从渲染插值姿态推导物理状态或用总速度假装接触冲量。
- [ ] GPU 批量求解软体接触；CPU 刚体侧只回读紧凑的逐刚体反馈，不为碰撞每帧回读/recook 完整表面 Mesh。同一步同步成本、接触迭代与物理阶段明确；延迟耦合不能冒充同一步耦合。
- [ ] 验证同高度下落时间、双方受撞速度变化、偏心撞击角速度、静置支撑、重置/禁用/销毁及重复 Play。数值测试检查无外力接触的动量交换，截图不代表双向反馈验收。
- [ ] 用户要求真正双向刚软体持续交互 **300 FPS+**，且按上面的刚体数量阶梯验证扩展性。使用真实编辑器 Play，固定分辨率、材质、网格规模、9 子步和物理频率，分别记录首次准备、落地、持续压挤、连续撞击的平均/P50/P95/最大帧时与 CPU/GPU/同步成本。不得用静止、减子步或关闭绘制代替。当前未完成、未达标。

## 8. A04：动态 Mesh、实例与通用计算资源

### A04.1 Mesh 主路径

网格工具双层入口（2026-09-09）：

2026-09-09 局部实现：既有 set_inline_mesh_data 支持 normals=None，在原生提交时按三角形面积累计法线；显式法线不被覆盖，分裂顶点不自动焊接，退化/孤立顶点法线为零。桌面 GPUJelly 已改用该入口，移除项目内每帧 np.add.at 法线累加。仍是 CPU mesh 提交，切线工具和 GPU 驻留直供渲染未完成。真实编辑器 240 帧短窗口热态/重新掉落/冲击平均 2.501/2.353/2.427 ms，P95 7.354/6.758/7.186 ms；不包含普通刚体接触，不视为持续交互 300 FPS 已达标。

2026-09-10 GPU/CPU 派生属性纵向切片：驻留 `inx.buffer` 已直接作为 canonical Vertex 流供绘制；任何本阶段写入该流的 kernel 在阶段末统一追加一次 GPU 法线/切线重建，不回读完整网格。法线和切线服从索引拓扑，已删除为 Sphere 按相同坐标猜测焊接 UV seam 的特例。CPU inline 路径新增显式/自动 tangent 输入、`get_tangents` 以及 `recalculate_normals/tangents/bounds`；显式属性不覆盖，MeshCollider 仍只在显式 `recook` 时更新。真实 Vulkan 覆盖变形、正交归一 tangent frame、split seam 与资源退休；精确 GPU Bounds 仍待接入 GPU 剔除，不用隐藏 readback 伪装自动完成。

2026-09-12 第二消费者实证：桌面 `Infernux041Lab/02_InteractiveSnow` 使用两个公开 `@inx.compute.kernel` 在驻留 canonical Vertex buffer 上直接压实和重置高度场，阶段末由同一通用 Mesh 路径重建法线/切线并直接绘制；正常帧没有 `get_data()`、整网格 CPU 回读或重新上传。真实 MCP 在 1920×1080 Game View 验证 128×128 与 256×256 两档，256×256 为 65,536 顶点，稳态显示约 `1195 FPS / 0.8 ms`、脚本记录单次压实 dispatch 约 `0.09 ms`；按键移动形成连续轨迹，`R` 恢复平面，保存、重新加载再 Play 后仍为 0 warning/0 error。该证据证明动态 Mesh/自动法线/计算到绘制的纵向链路可复用，但不替代实例、Texture、异步回读、marker 分项和 Player 导出验收，因此 B02/A04.2 组合项仍不提前勾选。

- [x] 保留 NumPy 顶点/索引/法线/切线手动编辑与提交，同时提供内建法线、切线、bounds 重计算，项目无需重复实现基础网格处理。手动法线不被提交隐式覆盖；重算法线与切线分别定义。
- [x] 法线按共享顶点拓扑累计并归一化；UV/硬边分裂顶点默认不猜测焊接。覆盖退化三角形、孤立顶点、空网格、分块/共享网格、手工法线与切线，以及发布后版本/bounds/多 Renderer 可见性。
- [x] 同一 Mesh 资源主干承载 CPU 编辑与 GPU 驻留计算；内建工具在相应数据所在地执行。GPU 顶点/法线直接供绘制使用，不在内部隐藏整网格 CPU readback/upload。外部数据以 inx.buffer 和标量为主，NumPy 为可选兼容接口，不公开 Taichi ndarray/field 或裸 VkBuffer。
- [x] 复用资产身份、几何代际、资源租约与任务依赖。渲染 Mesh 更新不暗中重建 MeshCollider；软体/刚体交互按 A03.4 耦合，不把渲染更新视作双向物理已经实现。

后续发布接入按 [Mesh 编辑发布接入表](041-mesh-publication-map.md) 推进：内存修改必须贯通 registry 版本、依赖通知、共享 Renderer 和 GPU 消费，不能把文件重新导入或不在注册表中的 GUID 当成运行时 Mesh 的实现。

2026-09-09 registry 接入局部：已有 update_mesh_positions(guid, first, NumPy positions)，保留共享实例并更新 runtime version/驻留/加载代际，发出运行时通知。两个 Renderer 与显式碰撞 recook 隔离的集成用例通过，扩大回归 553 项通过。仅支持位置区间、保留其它属性；新建/复制、完整属性编辑、实际 GPU 画面和保存/Player 尚未完成。

2026-09-09 区间更新局部：原生 InxMesh.UpdateVertexRange 沿同一几何发布路径更新，保持拓扑/材质槽并重算总体与分块 bounds，越界不发布。原生测试与 505 项回归通过。尚未公开 Python 可编辑资源入口，也未贯通 registry/GPU 更新，不等于局部上传性能完成。

2026-09-09 共享资源底座：InxMesh 发布不可变几何代际，SceneRenderExtractor 保活该几何而非可变容器。旧代际/复制隔离/空 bounds/最后引用退休的原生测试与 505 项扩大回归通过。公开可编辑 Mesh、共享资源版本向 registry/GPU 传播和区间更新仍未接完，不能据此勾选完整模型。

2026-09-09 局部推进：NumPy `set_inline_mesh_data` 复用已有程序化几何提交路径，不再隐式通知 MeshCollider 烘焙；新增公开 `MeshCollider.recook()`，沿现有 worker/物理发布边界重建。真实 native 与 wrapper 的可见网格抬升/射线碰撞旧面/显式重建新面测试通过。此变更不等于共享 Mesh 资源、区间更新、GPU-only 或碰撞烘焙失败保留旧形状已经完成。

2026-09-11 公开 Mesh 资源纵向切片：新增 `inx.Mesh.create/from_data/load/load_guid/copy`，运行时新建 Mesh 与导入 Mesh 同样进入 `AssetRegistry`，使用独立运行期身份、不可变几何代际、registry runtime version 与依赖通知。公开整块/等长区间顶点流发布、顶点/索引 NumPy buffer、submesh、材质槽和显式法线/切线重算；`MeshRenderer.mesh/shared_mesh` 直接读取共享资源，只有显式 `copy()` 才产生独立身份。两个 Renderer 共享可见、复制隔离、一次发布一次版本、失败发布原子回滚及既有资产/碰撞回归共 `58 passed, 1 skipped`，Windows Release 原生模块编译通过。布局容量/GPU-only 读取策略和完整运行时退休仍保持未完成。

2026-09-11 运行时 Mesh 生命周期收口：`inx.Mesh.destroy()` 只销毁明确的 `runtime-mesh:` 身份，沿现有依赖通知主干解除所有 Renderer/Play clone 引用；导入资产不能被该入口误删。真实 PlayMode 场景重建进入与退出均复用同一注册表资源，显式 `copy()` 保持资产隔离。Vulkan 删除测试在两个 Renderer 已经提交在途帧后销毁 Mesh，要求资源记录、待上传队列与该 GUID 驻留归零，并以 GPU completion retirement 释放旧 buffer，不等待设备空闲。完整 Windows Release CTest 为 `76/76`，其中 16 项使用真实 Vulkan 设备。

2026-09-11 GPU Vertex 容量局部接入：`MeshRenderer.create_vertex_buffer(capacity=...)` 现在显式建立 `float32 (capacity, 23)` 的 canonical Vertex 存储，当前 Mesh 的 `vertex_count` 是有效绘制范围，容量仅决定一次真实 Vulkan 分配；绑定允许 `capacity >= vertex_count`，但严格拒绝其它 shape/dtype。GPU 派生法线/切线只处理有效顶点和索引拓扑，测试证明额外 64 个槽在变形与重建后仍未被写入。公开 `vertex_buffer_capacity` 可直接观察实际分配。Python Mesh 集中回归 `23/23`、真实 Vulkan resident Mesh 回归 `1/1` 通过。索引容量、有效范围动态变更和 CPU-readable 实际释放仍未贯通，因此下面两项保持未勾选。

- [x] 将现有整块 inline 写入整理成正式 wrapper/stub，声明数组 dtype/shape、复制与提交时机。
- [x] 一个 Mesh 资源模型支持显式新建、共享、复制、整块/区间更新、submesh、材质槽、顶点/索引 buffer。
- [ ] 不因读取某个属性隐式克隆资源；布局/容量与本次有效元素数量分开，重分配时机明确。
- [ ] CPU 可读数据保留是显式配置；GPU-only 不用隐式回读满足读取。
- [x] bounds、法线、切线可由作者提交或显式重算，拓扑和姿态更新不悄悄触发 collider cooking。
- [x] 动态 collider recook 使用现有 cooking/提交边界，另有显式请求；失败不替换成无关碰撞形状。
- [x] 运行时 mesh 的销毁、Play clone、资产隔离、GPU in-flight 退休符合现有资源体系。

### A04.2 草与计算基础

2026-09-09 生命周期补充：公共 Instantiate 的碰撞源复制隔离、复制体独立 recook、确认在途后删除 owner 并创建替代对象的回归通过；扩大场景测试共 410 项通过。保存重载/Player 的源资源持久化语义仍需统一 Mesh 模型承接，不增加临时私有文件格式。

2026-09-09 快照补充：显式 recook 选定未缩放源几何，在途完成和后续 collider center/scale 使用同一来源，不重读逐帧可见网格；实际射线、作业数及请求替代测试通过。显式 SyncTransforms 已将缩放产生的 cooking 纳入等待边界。Clone/持久化/销毁与 GPU 数据一致性仍需完整验证。

2026-09-09 烘焙替换补充：复合碰撞不再跳过尚未完成或失败的子形状后提交残缺候选。单体/复合体在主线程与 worker 两类失败后保留旧网格、有效重建后替换的真实射线回归通过；仍需请求快照、取消与 GPU 碰撞数据发布验证，不勾选完整烘焙合同。

- [ ] 提供共享几何的批量实例提交、实例属性、自定义 bounds、可见性与部分更新；不以成千上万 Python 对象循环代替批量通路。
- [ ] 通用 buffer/texture、typed layout、kernel 任务提交、read/write 依赖、barrier、批量执行、indirect 消费按项目所需贯通。
- [ ] GPU 计算结果直接供 vertex/instance/texture 使用，正常帧不经过 NumPy 回读再上传。
- [ ] 异步 readback 明确完成通知、延迟、取消和资源所有权，不把每帧阻塞 wait 包装成 async。
- [ ] 计算可在固定步、帧级或离屏任务中提交，不依赖 Game Camera 存在；所有权仍属于引擎资源/任务调度，不新建 Taichi 专属世界。
- [ ] 复用并补 CPU/GPU marker/timestamp 的实际缺口，以计算、同步、上传、绘制分别计时。

验收：程序化平面、绳网、草实例和计算驱动 mesh；记录 CPU 提交、GPU 执行、复制、内存与容量变化。没有实测不声称零拷贝/固定性能。

果冻性能验收补充（2026-09-09）：用户对照的是编辑器内 Play，报告 Unity compute shader 同任务约 300 FPS，而当前 Infernux 刚开始约 5 FPS、稳定后约 70 FPS。这个实际交互性能问题未解决，不以私有探针、无源码加载成功或静止场景 FPS 代替验收。必须分开记录首次 Play、已准备后的重新掉落/冲击、持续运动、稳定阶段的真实帧时间 P50/P95/最大值、每帧固定步数量和各阶段耗时；排除其它测试 Player 的负载，不将编辑空闲限帧混入 Play 窗口。对照双方需核实同硬件、分辨率、物理子步/规模、材质及绘制范围，300 FPS（约 3.33 ms/帧）作为用户参考，不用减少物理质量或停止模拟制造达标数字。先验证 Taichi 生成的 SPIR-V 设备执行和宿主调度；若同条件下编译质量/调度仍不能满足目标，应作明确的优化或切换 compute shader 后端决策，而非持续交付未接入的局部原型。

果冻运动验收补充（2026-09-09）：除帧率外，还要对齐模拟与场景的长度单位、重力、固定步、碰撞接触和渲染插值。禁止用显示端整体放大位移但保留小尺寸动力学来冒充同尺度软体。桌面 Lab 的模拟/表面/Gizmo 已统一为米制 1.8 米演示体；材料与冲击按尺度关系重新创作，场景参数通过 MCP 修改并保存，未修改全局 time_scale。增加同算法 CPU/GPU 的自由落体、接触摩擦、体积/翻转、冲击后轨迹及表面插值验证。局部探针不等于真实刚体同场景对照或视觉验收，后两项仍需补齐。

插件更新验收补充（2026-09-09）：Python/资源更新应沿用现有资源刷新、卸载/preload 和贡献发布，在当前编辑器进程完成，不把重启作为日常更新步骤。已加载原生二进制的替换仍是未完成边界：不能仅删除 sys.modules 或尝试覆盖占用中的 DLL 来宣称热替换；需要单独明确原生模块代际与资源退休方案。引擎本体原生模块升级与插件 Python 内容更新不能混为一谈。

2026-09-12 Python 插件热更新实证：运行中的 `Infernux041Lab` 直接更新已安装 MCP 包的 `operation_support.py` 与 `scene_operations.py` 后，现有 preload/刷新路径在同一编辑器进程发布新操作 schema；随后默认层级查询立即采用 Mesh 摘要，显式完整查询仍可用，控制台 0 warning/0 error。该证据只完成 Python/声明内容的日常更新主路径，不宣称已加载原生动态库可以覆盖替换。

## 9. A05：Infernux Buffer、CPU JIT 与 GPU Compute

2026-09-09 最新决议：CPU 函数执行和 GPU 单 work-item 计算分开；新增引擎级 inx.buffer，set_data/get_data 表达传输，向量使用已有小写 vector2/vector3/vector4。废止共同 hpc(device=...)、强制 NumPy 和公开 resident/upload/download 作为最终方案；不保留 GPU kernel 自动 CPU 回退。

完整执行项、例子、编译规则和验收见 [Buffer / JIT / Compute 合同](041-buffer-jit-compute-contract.md)，B01—B06 全部纳入 041，不因使用附录而延期到 042。

### A05.1 引擎资源与作者模型

- [x] B01：inx.buffer 的类型/布局、CPU/GPU 存储、set_data/get_data、批量初始化、NumPy 可选交换、所有权和在途退休。（2026-09-12：完成公开不可变描述、dense 标量/vector2/3/4、显式同步回读与有序上传、无复制/只读 view、Mesh 共享持有、资源代际及 exact-ticket bind-group 退休；真实 Vulkan 覆盖非整工作组、byte-offset view 与容量重建。）
- [ ] B02：共用 RHI 资源底座，贯通动态 Mesh/法线/材质/实例数据；审计粒子、Gizmo、物理交换的 GPU 化收益，不假定所有机制迁 GPU 都更快。
- [x] B03：@inx.compute.kernel + compute.launch；index(buffer) 声明执行域，普通循环为线程内串行，引擎处理 dispatch；多全局阶段显式分多次 launch。（2026-09-12：完整公开合同、辅助函数、全局 atomic add、跨 launch barrier、尾部范围及错误边界均由真实 Windows Vulkan 回归覆盖；作者不填写 dim/group，不公开尚无消费者的 workgroup shared/barrier。）
- [x] CPU JIT 与 GPU Compute 共享数值类型/数据资源，不共享并行语义或错误重放；GPU 不支持时准确报告，显式 CPU 算法另行选择。（`inx.buffer`/vector 布局共用；CPU Buffer 传 GPU kernel 明确拒绝，能力、编译、执行错误分类；GPU kernel 不进入 CPU auto-parallel 或失败重放。）

### A05.2 CPU JIT 与 V8 级分析/优化

- [x] B04：@inx.jit.compile 普通调用；保留并增强现有自动串并行与跟踪，直接接 CPU buffer/向量及合法 NumPy。
- [ ] 引擎拥有类型反馈、控制流/数据流与别名分析、受控特化、拆箱/向量化/内联、低编译成本基线到热点优化版本的策略；复用 Numba CPU codegen，不只是改装饰器名。
- [ ] 类型/字段/函数依赖代际在发布边界失效，必要入口 guard 保留；执行后的错误不能从头回放为 Python/串行，不通过运行时安装或自修复维持路径。
- [ ] 低调用开销、冷编译、预热、稳态吞吐、P95、代码缓存与内存同测；与 CPython、旧 JIT、V8/Node 和原生参考逐工作负载对照。目标超越 V8 不等于预先宣称全面胜出。
- [ ] 函数内去优化/OSR 单独评估，不未经状态恢复合同承诺；041 必须交付附录规定的分析、特化、优化选择与测试，而非只保留未来研究标题。

### A05.3 Taichi 纯 Vulkan 编译模块与 wheel 内置

2026-09-09 直连里程碑：Taichi 编译产物已导出 SPIR-V 与具名 uniform/storage binding 元数据；Infernux ComputeKernel 支持稀疏 binding，真实 Vulkan 测试按 `inx.buffer` 实际长度派发并正确写回。该执行完全绕过 Taichi launch、设备数组和 command list，证明 Taichi runtime 可从最终热路径移除。随后新增无设备 compiler-only 上下文，独立进程不创建 Infernux/VkDevice/队列/显存即可产出 SPIR-V。当前仍通过临时上游 Python AST 与 `Program` 壳取得 lowered kernel，wheel 内只有私有原生模块和引擎 loader，尚缺可用的 Infernux 作者 frontend；不得将此项勾成完成。

- [ ] B05：输入引擎 kernel 数值代码/类型化 IR、buffer 布局及目标能力，输出 SPIR-V、工作组/参数绑定元数据和诊断；不持有活动设备、buffer 或执行状态。
- [ ] 移除 field/SNode/设备 ndarray 所有权、ti.init、Program 执行 runtime、独立 RHI/分配/提交/同步/回读/释放；必要纯编译 IR 保留，禁止仅改名继续维护旧运行时。
- [ ] 继续裁剪 Taichi CPU/LLVM、CUDA/Metal/OpenGL/DX、GUI 和无关入口；CPU Numba 的 llvmlite/LLVM 保留。单 work-item lowering 替代旧顶层 for 自动 offload。
- [ ] 将源码依赖从 external/plugins/infernux_taichi 迁至 external/taichi_for_infernux，保留 Git 历史/改动/署名，更新子模块、构建和 CI；CMake 直接产出引擎 wheel 正式载荷，不再生成 Taichi 插件作为交付。
- [ ] 删除 Taichi 插件 manifest/preload/provider、catalog 与项目安装要求；内置编译入口沿引擎启动/准备/关停，按需初始化但不在首次调用下载。迁移旧项目避免重复注册，不删除用户插件内容。
- [ ] 干净 Windows/manylinux wheel 无独立 Taichi 安装及无项目插件时编译可用；无 GPU 构建进程可生成目标产物。设备资源与执行始终归引擎，不能重新私建 runtime。
- [ ] 适用的 Player 与引擎 wheel 一起携带 CPU/GPU JIT 工具；当前不交付 AOT。依赖收集、代码输入、平台动态库与初始化规则见 B06，不在首次调用联网补编译器。

### A05.4 构建、平台与迁移验收

- [ ] B06：迁移旧 hpc/resident、Lab、测试、源码识别、cook、stub 和文档；Editor/Player 的 CPU/GPU 双 JIT 载荷与能力声明。
- [ ] CPU JIT 的发行能力逐平台实证；Windows/Linux、Android Vulkan JIT 与 WebGPU 待验证路径各自给出最终制品证据。不恢复 Taichi CPU/Wasm，不将桌面 Numba 当成浏览器支持。
- [ ] 适用 Player 与引擎安装载荷统一携带 CPU/GPU 两套 JIT 工具；依赖收集、Nuitka 字节码、动态库和受管代码输入按 B06 验收，不只改编译器输出目录。
- [ ] 保留 LICENSE/NOTICE/原作者及修改标记，依据最终引擎 wheel/平台 Runtime/Player 实际载荷核对，不因内置或裁剪丢失。
- [ ] 验证 CPU/GPU 原地数据语义、无隐式回读、计算→渲染依赖、代码与资源退休、无设备/缺编译能力/非法布局诊断、导出和热更新；A03.4 的真实交互性能门槛不降低。

2026-09-10 JIT-only 构建收敛：移除 BuildRequest、桌面/平台 Cook、Build Settings 和自动化宿主中的 GPU AOT 预准备、临时目录、artifact staging 与 `_aot` 私有装饰器注入。Player 只保留源码编译后的 `@inx.compute.kernel` 声明，首次 launch 由随引擎载荷分发的 compiler-only frontend 生成 SPIR-V，再交给 Infernux RHI；粒子图自身的离线 artifact 不属于此处 Taichi/Python kernel AOT，保持独立。此项只消除已经否决的双轨，不代表各平台 JIT 制品与干净安装验收完成。

2026-09-11 旧 compute 作者路径退役：删除公开 `compute.hpc`、`resident/upload/download/dispatch`、NumPy `batch`、旧 GPU provider 与未实现 `compute.nn` 占位入口，并同步 stub、候选脚本策略、源码识别、Player cook 与回归。CPU 只使用 `inx.jit.compile`，GPU 只使用 `inx.compute.kernel/index/launch`；关闭 JIT 载荷时 CPU JIT 声明直接报错，不保留 Python fallback。Lab 已无旧 API 引用，定向回归 383 passed、1 skipped。B06 仍需最终 wheel/Player、多平台、许可与体积证据，不能据此勾选总项。

历史实证边界：旧 hpc/NumPy/preload/AOT、私有及实验 resident、RHI queue 等局部测试已见执行记录，可作为迁移回归基线；它们不证明上述新 API 已实现。最近打包的旧编译器不包含所有工作区改动，未构建/未验证项不能写成已交付。

## 10. A06：统一 UI 布局、世界绘制与字体

2026-09-14 UI 性能专项（用户要求优先继续，尚未收口）：屏幕与世界 UI 均需独立统计布局/文字、输入映射与命中、依赖检查、命令重建和原生 GPU 提交。第一轮已删除内部 bookkeeping 写入时的旧值读取、缓存 Canvas 祖先查询，并将材质依赖检查与颜色/路径读取分开；真实 12 号场景六按钮及 03 号场景 Screen/World 文字修改、父级平移旋转在 Game/Scene 出图通过。局部 A/B 使用真实原生组件、12 层祖先、100/1000 控件：1000 屏幕 UI 的依赖检查 P50 从19.881ms降到11.374—11.423ms，世界 UI 从14.819ms降到13.113—13.222ms。**这些数值仍非常高，且仅为依赖检查 CPU 微基准，不是整帧或 GPU 数据，不能作为 UI 性能验收。** 回归同时修复了 Scene 原生地址复用导致的世界 UI 旧缓存问题，使用现有 world_id 区分所属世界；最终完整 Python 6217 passed / 11 skipped。下一步需要把逐控件 Python 几何/材质轮询变成由权威变更驱动的依赖更新，静态 UI 不展开矩阵/重新排版；改变一个控件不应重建所有屏幕/世界 UI。保留同帧 Transform、材质重导入、布局、透明排序、深度、裁剪和输入捕获语义；再以控件数量、动态比例和多相机矩阵核对真实编辑器及 Player。不要通过降帧或隐藏界面得到虚假提升。

### A06.1 Figma 式作者模型

- [x] 统一屏幕/世界控件、布局和交互；普通世界物体可添加文字/图片/按钮，无需先建特殊 Canvas。
- [x] Frame 是可选的组合布局容器，不是每个文字必须套的渲染分类。屏幕 UI 使用 Canvas/最近布局容器作为参考；世界 UI 不存在根参考矩形、画布范围或隐式承载面，Frame 只能显式排列自己的子项。
- [x] 支持水平/垂直 Auto Layout、Fixed/Hug/Fill、padding/gap/对齐、min/max、绝对定位子项与约束；内容裁剪只属于屏幕 UI，世界 UI 不继承 Frame/Canvas 裁剪。
- [x] 明确 Hug/Fill 循环依赖及过约束的作者规则；不靠多轮无限求解/猜测尺寸。
- [ ] 现有 x/y/rotation 通过一次明确迁移写入 GameObject 原生 Transform；width/height/anchor/Auto Layout 保留为 UI 布局数据。迁移结束后位置和旋转只有 Transform 一个运行时权威，不引入 HTML DOM、完整 CSS 或第二套 RectTransform。
- [x] 屏幕 UI 与世界 UI 共用 GameObject 原生 Transform。屏幕 UI 把本地 XY 解释为逻辑像素；世界 UI 使用引擎内部统一的逻辑像素换算，不向作者暴露 PPU、Canvas、根面或范围。对齐、±90°、UI Editor 拖拽、Scene Rect Tool、Inspector 和 Undo 都修改同一 Transform；UI scale 保持可访问但不参与 UI 几何。
- [x] 自动宽高、文本测量、裁剪和 hit test 使用同一布局结果；编辑器 zoom 只改变预览，不回写作者尺寸。
- [x] 增加独立 alpha/interactable/blocks_raycast 子树分组；透明不等于穿透，禁用时取消已捕获交互。
- [ ] 完成 ProgressBar/Slider、值变化事件与现有多触点拖拽；RawImage 直接引用受管理纹理/RenderTexture，不存跨帧临时 graph handle。
- [ ] UI 与普通 GameObject 的便捷交互保持 Unity API 边界：Screen UI 与 Canvas-free World UI 使用 `on_pointer_enter/exit/down/up/click`（对应 Unity UI EventSystem/`Button.onClick`），普通带 Collider 的 GameObject 使用 Unity `OnMouse*` 语义的 Python 风格 `on_mouse_enter/over/exit/down/drag/up/up_as_button`，不虚构 MonoBehaviour 的 `OnClick`、`WorldHover` 或 `WorldClick` API；按下/释放必须使用同一命中对象捕获规则。
- [ ] 为上述公共交互建立性能收口：静态 UI 不逐控件轮询布局/材质/Transform，布局和绘制命令按权威 revision 增量更新；单个控件、共享材质、父 Transform、动态文字和多相机分别测量布局、命中、Python 调度、原生提交与 GPU 时间，禁止以隐藏 UI 或降低帧率目标冒充优化。
- [ ] 指针便捷 API 与 Raycast 共同优化：保持 Unity `on_mouse_*` 与 UI pointer/click 的可观察顺序不变，建立按视口批量取样、layer/trigger 预过滤、候选排序和 pressed/hover 状态迁移的独立路径；评估将稳定的射线投影、候选收集和命中结果写入下沉到 C++，以及在不改变本帧输入语义时的异步/延迟查询边界。不得通过跳过射线、降低命中精度或把旧命中当 fallback 伪造提速；必须分别记录 Python 调度、原生查询、等待/同步和事件回调成本。

2026-09-14 值控件公共合同：`UIProgressBar` 与 `UISlider` 统一提供 `set_value(value, notify=True)`、`set_value_without_notify(value)` 和 `on_value_changed`；值始终先经过同一套范围裁剪，指针拖拽、脚本设值和静默恢复不再形成不同语义。定向 UI/材质/RenderTexture/世界输入回归 **280 passed / 2 skipped**。该进展不提前勾选整项，RawImage 的受管纹理引用、Inspector/Player 及完整多触点验收仍需继续。

2026-09-12 Figma 布局第一纵切：新增视觉中性的 `UIFrame`，其最近父 Frame 是子项唯一参考矩形；普通 UI 子项默认保持既有绝对布局，进入水平/垂直 Auto Layout 后可选择 Flow/Absolute、Fixed/Hug/Fill、padding、gap、交叉轴对齐、min/max 与 Fill 权重。Hug 父项与同轴 Fill 子项被作为确定性作者错误直接拒绝，不进行迭代猜测或 fallback。Hierarchy 与 UI Editor 均可直接创建 Frame，UI 创建保留 Canvas 子树中的所选父节点，不再强制扁平挂到 Canvas 根；Inspector 已暴露上述字段。Frame 的祖先裁剪矩形现在由同一布局结果求交，编辑器 ImDrawList、桌面 Vulkan Screen UI 和 WebGPU Screen UI 均提交真实 scissor，不再只有序列化开关。另新增独立 `UIGroup`，其 alpha 只乘视觉透明度，interactable 与 blocks_raycast 分别控制事务和遮挡；运行中禁用交互会以 canceled 事件释放已捕获 pointer/drag。算法、嵌套创建和编辑器合同回归为 260 passed、1 skipped，交互/裁剪定向回归为 23 passed。约束表达、统一文本测量/hit test 和世界空间渲染尚未完成，因此不勾选对应总项。

2026-09-12 Auto Layout 主轴收口：`UIFrame` 新增稳定的 Start/Center/End/SpaceBetween 主轴排列，和既有 Start/Center/End/Stretch 交叉轴语义分离。多个 Fill 子项不再按初次权重份额各自 clamp 后留下空洞；确定性约束分配器按作者顺序固定触及 min/max 的子项，再把剩余空间按权重分给仍可伸展的子项，约束总量超过容器时允许明确溢出而不猜测缩放。真实 041Lab Game render-target 的上排三个 180px 子项跨 1200px Frame 均匀分布；下排前两项分别受 150/250px 上限约束，第三项占满剩余空间。定向回归 `152 passed, 1 skipped`，真实 Console 0 warning / 0 error；临时对象已丢弃。至此水平/垂直、Fixed/Hug/Fill、padding/gap、双轴对齐、min/max、Absolute、锚点约束和内容裁剪组成一条主路径，本项完成。统一迁移、Transform 组合和完整控件项仍单独未勾选。

2026-09-12 UIText 单一布局权威：删除 UI Editor、Inspector 与 Runtime Renderer 三处各自测量并回写作者 `width/height` 的路径。Auto Width/Auto Height 现在在绘制和命中前通过同一个预布局入口生成瞬态逻辑尺寸；运行时按 Canvas 缩放测量后归一回逻辑像素，固定宽度不再附加经验容差。`get_rect`、UIFrame、裁剪、绘制和 Canvas raycast 全部消费该派生尺寸，预览 zoom 不会污染场景文档。真实 041Lab Game render-target 显示了 Auto Width 以及 260px 宽的中文/Latin Auto Height；运行后场景文档仍保留作者的 520×42 和 260×12，而画面使用派生框。定向回归 `155 passed, 1 skipped`，真实 Console 0 warning / 0 error。本项完成。

### A06.2 世界空间按三维物体处理

- [x] 每个控件自身的局部二维几何通过普通 Transform/Camera 放到三维空间；复用 ImDrawList/字形基础，不原样提交屏幕 draw data，也不先聚合到世界画布或面板范围。
- [x] 默认深度测试、场景遮挡、相机剔除和世界变换；透明绘制继续遵守正常透明规则，不能声称 depth 取代全部排序。
- [x] 字形/图片的透明空白不写成实心矩形遮挡；世界 UI 不接受祖先 Frame/Canvas 的裁剪或 mask，控件只按自身真实几何绘制与命中。
- [x] 空间深度决定世界控件关系；完全共面时只用稳定的场景/层级顺序消除闪烁，不以全局 UI sorting order 代替三维关系。
- [x] 默认点击尊重前景遮挡；射线映射局部二维后走现有 pointer/焦点/拖拽，捕获期间离开面板仍可连续拖动。
- [ ] billboard、固定世界尺寸/恒定屏幕字号、忽略特定关联对象遮挡和特殊置顶为显式策略，不改默认深度规则。
- [x] 普通文字/按钮直接绘制，不为每个元素分配 RenderTexture；完整界面显示在显示屏等场景才显式使用离屏输出。
- [x] 世界 UI 的选择、复制、Prefab、保存、Play clone 和 Rect Tool 沿普通场景工作流；每个控件都按自身几何参与 Scene 点击，不依赖根面命中。
- [ ] UI 使用普通材质资产而非只支持组件私有颜色：默认 UI 材质承接现有填充/贴图；文字、图片等各有一个材质槽，Button 明确提供背景与文字两个材质槽；自定义 UI shader 的属性、透明和深度合同完成真实 Inspector/Player 验收。

2026-09-12 世界绘制第一纵切：`ScreenUIList.World` 使用独立三维顶点流，作者仍复用现有 ImDrawList/字形与控件渲染器；每个 Canvas-free UI 根只声明一次局部像素原点、pixels-per-unit 和普通 `Transform.local_to_world_matrix`，不为每个元素分配 RenderTexture。RenderGraph 在 Camera UI 前加入显式 `_WorldUI` pass，复用当前颜色/深度附件并用当前 Camera 的 `Projection × View` 变换。041Lab 中通过 MCP 在已有屏幕 Canvas 的场景里将 Frame/Button 明确挂到普通 Empty 下，确认作者服务不再偷改父节点；真实 Game render-target 可见文字和按钮随 Transform 平移、缩放、旋转，移动到不透明刚体后方后被深度遮挡，控制台 0 warning / 0 error。Windows Release 原生模块编译通过，UI/RenderGraph 定向回归 101 项、作者父级回归 25 项通过。此证据只勾选局部二维→世界几何这一完整条目；世界交互、旋转裁剪、透明排序、剔除、Prefab/保存、多平台仍保持未完成。

2026-09-12 世界输入旧纵切（已被无范围规则取代）：原实现曾把 Camera 射线映射到 Canvas-free 根平面，并允许 Frame 裁剪后代。该“根面/局部裁剪”模型不再是验收目标；后续输入改为逐控件自身 Transform/几何命中，世界 UI 永不继承祖先裁剪。

2026-09-12 世界交互收口：`UISlider` 不再读取 `UICanvas.reference_width/reference_height`，而是使用每次 `PointerEventData` 已携带的 `canvas_size`；同一实现因此同时适用于屏幕 Canvas 与 Canvas-free 世界输入表面，不增加伪 Canvas 字段或兼容回退。定向测试 `128 passed, 1 skipped`。真实 041Lab 中 MCP 创建 300×50 世界 Slider，通过 Game View 在左端按下、拖到控件右侧之外并保持按下，运行时值由 0 更新为 100，render-target 从空轨道变为完整填充；前景遮挡、旋转射线、Hover/焦点/拖拽分发与捕获期间原表面坐标至此组成同一条实证链。临时对象已丢弃，控制台 0 warning / 0 error。

### A06.3 字号与文字专项

2026-09-12 编辑器高 DPI 反馈：**按用户决定在 041 阶段性收口，不将尚未取得的其它设备矩阵作为本轮阻塞项**。已修正系统 display scale 与窗口 pixel density 的职责分离、显示缩放事件的 GUI 更新、真实 framebuffer 尺寸发布，并让 MCP 显式提供已有坐标映射而不为旧版本编造倍率。3072×1980、100%/125%/150%/200% 的 ImGui 绘制几何/命中合同测试与窗口策略共 4 项原生测试通过，相关 Python 50 项通过；这些最初是在 1920×1080、100% 主机上的模型测试，不是第三方原机器复现。用户提供的缩小截图只证明菜单/Hierarchy 可见而中央及其它面板异常，不能据此确定鼠标偏移倍率。

用户随后更换显示屏，本轮已取得 **Windows 2560×1440、150%（HWND 144 DPI、PMv2）实机证据**。同进程恢复窗口时，SDL/ImGui 已正确发布 1.5 倍，但字体仍停留旧字号：确认 ImGui 1.92 的 atlas Clear 会写回旧 FontSizeBase；重载字体后现显式发布新字号，菜单/工具栏的局部主题尺寸只缩放一次，专用固定工具栏高度随字体与 padding 更新，不重建用户布局。文件字体恢复 atlas 所有权，避免每次重载遗留文件内存。更新后 Release 构建和增强的同 context 字体/自动尺寸按钮回归通过；真实 150% 冷启动的完整编辑器已人工查看，菜单 57×33、播放按钮 57×39。真实 OS 鼠标在最大化 client 2560×1334 与缩小 client 1298×804、非零窗口原点下均成功切换 Console Follow 并恢复原值，坐标误差不超过 0.5px；项目菜单、Camera 弹层及 Hierarchy 选择同样成功。证据见 `dev/041-dpi-150-*.log` 与引擎侧 `041-dpi-150-cold.png`。原机器 3072×1980 的完整异常、运行中混合 DPI 跨屏，以及下方多平台文字矩阵仍未被这些结果替代，保持原未完成边界。

同一 150% 编辑器进一步通过真实 OS 输入完成 Inspector“添加组件”弹窗打开/关闭（未新增组件）、Scene 点击选中 Emitter 12、Game 页签与分辨率下拉，以及 Play→Stop。最终最大化 Game 编辑器截图已人工查看，项目 `dirty=false`，Console 0 warning / 0 error；验收未改系统缩放，也未使用隐藏 Player。`dev/041-dpi-150-os-game.log`、`dev/041-dpi-150-os-inspector-scene2.log` 保留操作证据。

- [x] 统一作者 font_size 的逻辑像素/字体度量定义、baseline、行高和字距；字体缩放、DPI、Canvas/世界比例和编辑器 zoom 各计算一次。
- [x] 验证 stb ScaleForPixelHeight 与 em 度量差异、字体替换、测量和渲染缩放链；这些是候选原因，不预判根因。
- [x] 用同一字体文件/字重，对齐 Figma 100% 的多字号、中英文混排、标点、多行/换行和字框；保存测量数据与参考图。
- [ ] 覆盖 Windows/Linux、Editor UI 预览、Game View、Player、多个 DPI/窗口比例，并对 Android/Web 公共文字样本复验。
- [x] 检查文字尺寸、基线、行距、裁剪与 hit rect，而非只看肉眼字高；不得用全局经验倍率修正所有字体。
- [x] 明确主字体和显式字体 fallback 链；已选字体缺失不可静默替成另一个字体后将差异算通过。
- [ ] 对世界缩放/透视下的清晰度做样本比较，再决定复用 atlas 是否足够或需要 SDF/MSDF；只维护选定主实现，不预先并列塞多种渲染器。
- [ ] 中文字体、纹理和文本在 Cook 包中按 GUID/路径映射正确读取，无需依赖开发机字体安装。

2026-09-12 字号语义纵切：同一 PingFangSC-Regular 文件、同一中英文/数字/全角标点字串在浏览器 CSS/Figma 风格参考页和 041Lab 原生 Game render-target 中按 12/16/22/32/48/72px 对照。旧路径在 22/32/48px 只有参考宽度的约 66.3%—67.0%；确认根因是 stb `ScaleForPixelHeight` 用 hhea 上升部+下降部解释尺寸，而 CSS/Figma 使用 em 方框。统一 `InxTextLayout` 现在直接读取字体 SFNT `head/hhea` 表计算每字体的 em→stb 换算，不使用全局经验倍率；逻辑 baseline/行高仍以作者字号计算，只有字形栅格尺寸发生换算。修复后 22/32/48/72px 的可见字形宽度为引擎 268/391/587/881px、参考 269/391/587/882px。原生解析测试、Windows Release 模块、UI 定向回归 `142 passed, 1 skipped` 均通过，真实 Console 为 0 warning / 0 error。多行/换行/字框、字体替换、DPI/平台矩阵尚未完成，因此只勾选统一字号定义这一项。

2026-09-12 多行/换行/边界纵切：同一参考页和 041Lab Game render-target 增加显式三行、固定 520px 的中文/Latin 自动换行、4px 字距和 36px 高越界裁剪。32px、`line_height=1.4` 的相邻行起点间隔为 45px，22px、`line_height=1.3` 为 29px；固定宽度中英混排在两端均从“定宽度”处换行，字距与字框一致。此前只存在于 Inspector 的 `TextOverflow.Clip` 现直接进入编辑器 ImDrawList 与桌面 Vulkan Screen/World UI 的共享文本提交，真实截图只保留第一行且第二行没有越界泄露。证据为 `out/041-font-css-reference.png` 与 MCP 会话 `review/font-metrics-engine.png`；原生 `infernux.text_layout`、定向 Python `143 passed, 1 skipped`、真实 Console 0 warning / 0 error。该矩阵条目至此完成；字体替换/fallback、hit rect、DPI/平台和世界透视清晰度继续保持未勾选。

2026-09-12 字体替换与 atlas 生命周期纵切：同一参考矩阵增加 Roboto-Medium 32px 样本；浏览器与引擎可见字形宽度均为 431px，证明每字体 SFNT 度量进入统一测量/渲染缩放链而非只适配 PingFang。DPI 引发的 atlas 重建现在先清空 `InxTextLayout` 自定义字体与缺失字体缓存，避免缓存持有 `io.Fonts->Clear()` 已销毁的 `ImFont`。Windows Release 原生模块和真实 041Lab 渲染通过，Console 0 warning / 0 error。显式 fallback/缺字链、字体 GUID Cook 和跨平台证据仍是独立未完成项。

2026-09-12 显式缺失字体边界：空字体路径是唯一选择引擎默认字体的方式；作者指定的非空路径无法解析、文件不存在或字体损坏时，原生文字布局现在返回空字体并对该路径只记录一次错误，不再静默替换成默认字体。`infernux.text_layout` 原生测试覆盖重复请求只报一次；真实 041Lab 为 Auto Width 样本指定不存在的字体后，该样本不再以默认字体冒充成功，终端只出现一次明确原生错误。临时样本已清理，场景 Console 恢复 0 warning / 0 error。逐字形的显式 fallback 链仍未实现，因此组合总项保持未勾选。

2026-09-12 显式 fallback 链收口：`UIText` 与 `UIButton` 增加作者有序备用字体资产列表，UI Editor、测量、换行和 Screen/World GPU 绘制在每个字形处使用同一个“主字体→显式备用字体”顺序；不同字体各自保留 SFNT em 换算。链中任一显式路径为空、缺失或损坏时整段布局失败并明确报错，不跳过错误继续找环境字体；所有字体都缺少某字形时只使用主字体 replacement glyph。原生测试以 Roboto 主字体和 PingFangSC 备用字体证明 Latin/中文分别命中预期 face，并覆盖缺失链；Python UI/Inspector/RenderGraph 集中回归 `286 passed, 1 skipped`。真实 041Lab Game render-target 显示 `Roboto primary + 中文备用字体 0123`，Play clone 文档保留有序列表，Console 0 warning / 0 error，临时对象已清理。本项完成。

2026-09-12 世界 UI 透明覆盖修正：世界 UI fragment 主路径在写颜色/深度前丢弃 alpha 为零的字形和图片空白 texel，避免透明字形四边形仍发布一整块深度；抗锯齿的部分覆盖像素继续按正常透明规则绘制。世界 UI 不再规划祖先 mask；自身形状透明与未来材质透明继续走正常材质合同。

2026-09-12 世界 UI 永久样本与坐标主链收口：041Lab 新增并保存 `Assets/Scenes/03_WorldSpaceUI.scene`，由普通场景对象直接承载 `UIFrame`、`UIImage`、`UIText` 和 `UIButton`，没有 Canvas、RenderTexture 或项目私有渲染脚本。修正世界顶点局部 X 轴与相机正面约定，并让射线到局部二维的输入映射使用同一符号；正尺度、零旋转的面板因此在 Scene 与 Game 都保持左到右显示，不再靠负缩放翻面。重启真实编辑器后，MCP 引擎 render-target 已分别捕获 Scene、Game 和真实按下态；Game View 的按钮由亮青色切换为 pressed 青绿色，证明显示与点击落在同一位置。Windows Release `_Infernux` 编译通过，Transform/MCP、世界 UI 与 Rect 路由集中回归 `40 passed`，Console 0 warning / 0 error。普通 Text/Button 的世界绘制明确直接进入共享三维 UI 顶点流，不为元素分配离屏纹理，故勾选对应条目；Prefab/复制/完整 Rect 生命周期、mask 与特殊 billboard 策略仍未完成。

2026-09-12 世界 UI 普通场景生命周期纵切：Canvas-free Frame/Button 可按普通对象 clone、保存 Prefab 和 instantiate，不会补建 `UICanvas`；Transform、组件自身尺寸、文字和子树保持。旧实证仍使用了后来删除的作者 PPU 字段，因此 Scene 点击选择与完整 Rect 生命周期在新无范围模型下重新打开验收。

2026-09-12 Transform 权威规则修订：放弃此前“UI x/y 与 Transform 单向组合”的中间方案。所有 UI GameObject 与普通场景物体共用同一个原生 Transform；隐藏的 x/y/rotation 只用于旧文档一次性迁移和兼容序列化，不再承担 Inspector、UI Editor、Rect Tool 或 Undo 的编辑权威。UI Editor 的拖动、旋转和带位移 resize 分别记录原生 local_position/local_euler_angles 与 UI 自身尺寸；世界 UI 不再存在根参考框，也不响应 Frame Clip/Mask。

2026-09-12 世界 UI 无范围主链：删除可序列化 `world_pixels_per_unit`、世界 root 查询、根对象 Rect 特判和 Rect Tool 的根范围分支。Canvas-free UI 逐控件以自身普通 Transform、width/height 几何、深度和射线命中独立提交；父 Frame 不再是渲染面、输入面或裁剪边界。原生接口也由 world surface 改成 world element，固定逻辑像素换算归引擎实现且不接受作者参数。041Lab 冷重启后 Scene/Game 均立即显示 Frame/Image/Text/Button；把 Title 移到旧 640×360 Frame 外，Game render-target 仍完整显示，恢复后场景 `dirty=false`。随后先在 Hierarchy 选中 Cube，再通过真实 Scene 画面点击按钮，选择切换到 `World Action`，Scene render-target 同时显示准确的选择边框和 Gizmo；Prefab/clone/保存/Rect 既有回归仍通过。Button Inspector 真实暴露独立背景材质与文字材质槽，序列化文档不含世界范围字段。Windows Release 原生模块构建通过，定向回归 `261 passed, 1 skipped`。

2026-09-12 世界 UI Scene 拾取仲裁修正：编辑器逐元素射线命中明确忽略运行时 `raycast_target/blocks_raycast`，因此装饰性 Text/Image 仍可直接选中；视觉中性的 `UIFrame` 不再用布局矩形抢走可见子项点击。Canvas-free UI 的精确 CPU 命中现在直接作为该次 Scene 点击的权威结果，不再被一帧后且无法编码文字字形的普通 GPU object-ID 回读覆盖；原本只服务粒子 Gizmo 的“保持当前选择”也收窄回粒子，不允许已选背景在新点击位置压住前景文字。真实 041Lab 冷重启后依次点击背景、再点击标题并等待异步回读，Hierarchy 稳定从 `Panel Backdrop` 切换至 `World Title`，场景保持 `dirty=false`；集中回归 `151 passed, 1 skipped`。

验收：手牌布局、自适应按钮、可旋转可遮挡的世界 UI、恒定字号标签、文本比较矩阵和用户交互均成立。

## 11. A07：所有场景物体可用的 Rect Tool 与 Handle

- [ ] Rect Tool 对所有场景物体可用，不只 UI；复用现有 Gizmo 拾取、坐标空间、吸附、多选和 Undo。
- [x] UI 拖动改变布局矩形，不默认拉扁字体；Fill/Hug 手动改变时按明确规则切 Fixed 或编辑约束，Inspector 同步显示。
- [x] 普通三维对象依据操作平面和包围范围修改 Transform 位置/缩放，不自动编辑顶点或烘焙模型。
- [x] 空物体、Camera、Light 不因缺 Renderer 禁用工具；以 Transform/子层级和定义明确的编辑参考范围操作，不新增虚假的资产宽高。
- [ ] 多选、父子同时选择、局部/世界、旋转对象、负缩放、不可缩放/锁定字段的行为明确；零尺寸不产生无穷缩放。
- [ ] 一次拖拽一次事务；Rect 的布局字段或 Transform 修改进入同一属性服务、Prefab override 和 Undo。
- [ ] Scene View 与 UI 编辑器复用矩形操作语义；视觉手柄可适配各自视图，不复制两套数据写入。
- [ ] 项目自定义位置/方向/半径/限位 Handle 经稳定公共入口注册；已有可用能力只补公开合同。
- [ ] 拖拽时对象删除、切场景、脚本重载、取消/失焦与插件禁用复用生命周期；测试后修真实缺口，不额外造恢复调度器。

验收：UI、mesh、无 Renderer 对象、父层级、多选均可操作；Undo/Redo 恢复值与选择语义一致，重开后结果保留。

2026-09-12 Rect/UI 布局旧纵切（无范围规则前的历史实证）：UI Editor 与 Scene Rect 共用 `prepare_layout_resize` 权威规则；被直接操作的轴从 Fill/Hug 切为 Fixed，UIText 同时切到 FixedSize，拖拽只写 `x/y/width/height` 等布局字段而不缩放字体或 Transform。旧实现曾从根 Transform 和作者 PPU 产生绘制/拾取 frame；该根范围与作者 PPU 已被删除。当前 Canvas-free UI 按每个控件自身 Transform 和真实几何登记 Rect，内部固定逻辑像素换算不暴露给玩家。普通 Mesh 和无 Renderer 对象沿统一 Transform 路径；多选/父子、Prefab override、负缩放、生命周期和公共 Handle尚未全矩阵验证，因此对应条目保持未勾选。

同轮普通 3D 实证：真实 041Lab 的旋转内建 Cube 由 Rect 读取 MeshRenderer 包围范围与当前操作平面；角点拖拽只把 Transform 从 `(-1.40, 0.16, -0.45) / (0.80, 0.38, 0.80)` 改为 `(-1.87, 0.03, -0.45) / (1.38, 0.64, 0.93)`，内建 `Cube` 网格身份与文件材质 GUID 均未变化。Ctrl+Z 精确恢复完整位置/缩放，控制台 0 warning / 0 error，故普通三维对象条目闭环。

同轮无 Renderer 实证：`ResolveRectFrame` 对 Empty、Camera、Light 使用明确的 Transform 编辑参考框，不创建或序列化资产尺寸。原生回归证明无 Renderer 对象仍可获得有效 frame 并提交完整 12 条 Rect 绘制线；真实可见 041Lab 中选中 `Key Light`、按 `T` 后，Scene View 在灯光 Transform 处显示可拾取 Rect frame，Inspector 中 Light 参数未被改写，Console 0 warning / 0 error。证据为 MCP 引擎目标截图 `review/041-rect-light.png`。

### A07.1 Gizmo 渲染路径性能（新增收口项）

2026-09-10 驻留线框纵向切片：`Gizmos.draw_lines` 已同时接受 NumPy 与 GPU `inx.buffer`。GPU 路径不再为 Scene Gizmo 调用 `get_data()`，而是在 Gizmo 收集阶段以引擎内建 kernel 生成 canonical Vertex 流，渲染直接绑定同一驻留缓冲；索引拓扑按稳定身份首帧上传，后续不逐帧哈希、比较或重新上传。新增 `draw_wire_spheres`，共享一份单位球拓扑并在 GPU 展开所有中心，替代逐节点 Python 建模。041Lab 的约束线和内部节点均已迁移，可见编辑器冷重启后节点随软体运动、控制台无警告/错误；真实 Vulkan 覆盖驻留线与球、Mesh 发布和资源退休。2026-09-11 又将普通 immediate-mode Gizmo 中相邻、同 world matrix 的线批次直接合并为一个原生 draw descriptor；颜色已在顶点中表达，因此无需按每次 helper 调用重新拆分，矩阵变化仍保持明确边界。尚缺回调/打包/上传/绘制分段计时及普通组件大负载对照，因此本节保持未完成。

- [ ] 以果冻内部节点/约束为真实负载，分别测量回调收集、Python 几何构造/分配、打包、GPU 上传和绘制耗时；记录节点数、线段数、批次数和帧时间，不能用减少节点或隐藏 Gizmo 代替优化。
- [ ] 提供 inx.buffer 批量点/线输入并保留 NumPy 兼容，合并相同状态的提交；共享线框球等单位几何，避免每个节点每帧重复生成拓扑、复制矩阵和创建小列表。
- [ ] 复用上传缓冲及容量，按实际动态数据更新；静态几何和动态位置区分生命周期，不引入逐帧内容哈希或新的恢复机制。
- [ ] 检查不可见 Scene、Gizmo 关闭与组件禁用时的工作量；保留世界变换、深度遮挡、颜色、拾取和帧资源退休的正确语义。
- [ ] 同一项目、同一相机和同一负载对比优化前后；覆盖普通组件、自定义大量节点/连线和开关切换，确认不是把 CPU 卡顿转移成 GPU 等待。性能预算在取得基线后确定。

## 12. A08：材质与 Renderer 参数覆盖

- [x] 材质默认参数和单个 Renderer 覆盖分开；不通过修改共享材质改变所有对象，也不隐式克隆材质。
- [ ] 正式公共参数接口支持 shader 声明的标量/向量/颜色/矩阵/纹理及必要数组/buffer 引用，类型来自 shader 反射。
- [x] 支持材质槽/实例归属、单字段移除覆盖和整体清空；多个系统不因覆盖某个字段误删对方其它字段。（2026-09-12：持久作者值与按稳定 owner 隔离的运行时层进入同一不可变参数块；同字段按实际写入顺序生效，移除/清空只影响指定 owner 并显露上一层。双 submesh、复制隔离、纹理 GUID 边界及相机剔除缓存已有集中回归。）
- [x] Renderer 覆盖只改变参数，不冒充改变 shader、blend、depth 等 pipeline state；这些仍通过明确材质/pass 配置处理。
- [x] 明确材质、实例、World 环境、View/pass 参数来源与优先级，不引入所有命名空间依次 fallback 的字典链。（2026-09-12：材质/Renderer 使用 set 0，World 环境先解析进各 RenderView 快照并使用 set 1，引擎帧/实例状态使用 set 2，pass-local 使用图参数块或 push payload，bindless 表使用 set 3；删除从未绑定却宣称实现的 `set_global_*` 临时字典 API。）
- [ ] 自定义 pass 能为指定 draw 绑定短生命期 payload，命令捕获提交时的值，不所有 draw 最终读到同一份被反复修改的数据。
- [ ] 保留批处理/实例化数据路径，按更新频率上传；衡量是否破坏批处理，不照搬 Unity MPB 后宣称必然更快。
- [ ] 序列化的实例覆盖、运行时临时覆盖、Play 隔离与纹理资源退休各有明确归属。

验收：三个物体共享材质，只改一个颜色；草的实例风参数；多 submesh；相机/反射 View 参数不串扰。

2026-09-12 Renderer 参数覆盖纵向切片：`MeshRenderer` 现在分别持有可序列化实例覆盖和按稳定 owner 隔离的运行时覆盖；同字段由最后实际写入的运行时 owner 生效，删除该 owner 后显露上一运行时层或持久值，整体清空不影响其它 owner，始终不修改或克隆共享材质。公开接口只接受当前 shader 反射出的参数，不允许借覆盖接口替换 shader 或改变 blend/depth/pipeline state。参数块以不可变快照进入 draw payload，Vulkan 为实际需要覆盖的 Renderer 绑定对应 UBO/纹理描述符；纹理覆盖在赋值边界验证 GUID，并进入 Renderer 运行时依赖集合。真实可见编辑器中三个对象共享 DefaultLit，分别显示继承白、运行时红、运行时绿；移除绿色覆盖后即时恢复白色，插件 Python 模块热更新和参数修改均无需重启，Console 为 0 warning / 0 error。过程中修复了相机剔除缓存只更新蒙皮、不更新参数块的通用 draw-payload 缓存错误；双 submesh 使用同一材质但独立参数块的原生回归也已贯通。标量/向量/颜色/矩阵/纹理和显式参数域已形成主链；数组/buffer、短生命期 pass payload、批处理性能和完整资源退休矩阵尚未完成，相关条目不提前勾选。

2026-09-12 单 draw 捕获底座：新增 `DrawParameterBlock` 与 `CommandBuffer.draw_mesh` 的首段公共接口；填写对象可复用，但录制 draw 时必须通过目标材质的反射类型校验并产生不可变值快照，随后修改原对象不会污染已录制命令。Mesh 几何代际、材质和值快照均由命令持有；`execute_command_buffer` 同样在调用边界复制命令，不再依赖 Python 对象活到帧末。显式 draw 已进入统一 `DrawCall`、对象缓冲和 Vulkan 材质描述符路径，原生测试验证捕获隔离、几何代际所有权和错误类型拒绝。当前只开放材质 primary pass；RenderGraph 内“指定 pass/指定 draw”、buffer/数组和批处理实测仍未闭环，因此上方两项继续保持未勾选。

## 13. A09：一个 RenderTexture 和通用管线能力

2026-09-13 跨队列重放续进：图内资源保留末尾所有者，下一轮由既有 pre-setup 路径先释放、首个消费者再获取；首次执行不增加释放。所有权跨族等待覆盖交接本身，同族数据依赖保持原有范围。颜色/Buffer 的9种队列组合与深度的2种实际采样路径完成120次读回，另覆盖 move/recompile/reset，同步验证通过。完整 Release、84/84原生、6143通过/13跳过Python，以及可见编辑器的800×450目标 Edit/Play/Stop 和正常退出均通过；不据此勾选下方包含按需调度、双缓冲等的完整条目。没有新建管理器、CPU等待或旧帧兜底。

2026-09-13 平面反射续进：041Lab 新增可保存的 `10_PlanarReflectionGallery`。普通 Camera、导入的 RenderTexture、文件材质和 Renderer 参数覆盖组成镜面，不新增专用反射渲染器。相机 NumPy 4×4 矩阵可直接交给材质/Renderer/单 draw 参数；编辑模式 Update 后执行 LateUpdate，使未 Play 时也有预览。真实按钮可暂停环绕、开关斜近裁剪，已检查镜面平移/倾斜和排除镜面层的非递归边界。两次 Play→对象禁用/启用→Stop→重开均通过，最终 Console 0 warning/0 error，引擎侧三路截图已目视核对。两个 240 帧窗口完整帧均值 1.336/1.344 ms、P95 2.121/2.108 ms，仅代表该 Windows 演示，不是果冻/刚体或整引擎性能验收。发现并修复快照漏掉 native-only 引用、字段绕过引用缓存失效、重复清缓存及旧清理回调解析到新世界的问题；旧 owner 的直接 `game_object` 清理访问仍不据此宣称已对齐 Unity。正式 Player/其它平台及本节剩余资源合同尚待验证，整项勾选不变。

2026-09-13 最新状态：`.rendertexture` 的 Project 创建、Inspector/Undo、Camera／材质／UIImage 持久引用、共享 GPU owner 与重导入已贯通；同一公开资源保存具体类型和 GUID，CPU 反序列化不分配 GPU。Importer 产出 `Library/Artifacts/RenderTexture/<guid>.inxrtex` 二进制描述，不保存像素；Camera 在渲染集合发布时解析目标，缺失引用不会偷偷变成屏幕输出。插件资产与 Assets 使用同一 Cook 选择与依赖主链。

正式 Windows 交付证据：CMake `prebuild_player_runtime` 直接产出平台插件载荷，测试项目通过普通插件更新流程安装；MCP 构建 Debug Player 并真实打开窗口。同一 HDR／4×MSAA／517×291 目标由 Camera 写入，Mesh 材质、UIImage 直接纹理槽和 UIImage 材质同时取样。通过 Project 历史删除临时场景／材质／纹理源文件后，原包冷启动仍正确显示，引擎捕获已目视核对。Content.inxpkg 包含 GUID 寻址制品，AssetCatalog 保留场景→材质→纹理依赖；磁盘输出没有裸露的 Assets／Library／ProjectSettings／Packages。此处是二进制容器与编译描述，不宣称不可提取的密码学加密。原09场景、Build Settings已恢复，两次测试Player均正常退出，Console0警告/0错误。

下方未勾选的完整条目仍需各自出口证据。Windows 作者与 Player 路径成立不等于 Linux／Android／Web 全部验收，也不等于独立 View history、平面反射和全部管线资源接口已完成。详细过程、测试及旧阶段限制见执行记录。对照 Unity 的 [Render Texture 作者流程](https://docs.unity3d.com/6000.0/Documentation/Manual/output-to-render-texture.html)，继续完成同一资源从编辑到发布的主链。

- [x] 公共只有一个 RenderTexture 资源类型，可被 Camera、材质、UI 引用；不并列增加 CustomRenderTexture 等作者必须选择的资源种类。（2026-09-12：同一运行时资源已接通 Camera.target_texture、Material.set_texture 与 UIImage.texture；真实 Windows 041Lab 的世界/屏幕 UI、UI 材质取样、resize 与替换已验收。资源资产编辑/导出和其它平台仍由下方相应条目验收，不据此勾掉整个 A09。）
- [x] 描述尺寸/相对尺寸、格式/HDR、采样、深度及用途；没有“最高质量统一格式”默认替代所有需求。（2026-09-13：固定像素与 Game 渲染像素比例二选一；同一公开 owner、原生批次 prepare/publish、Camera/材质/UI 实际联动通过。深度取样/storage 均显式声明，非法组合或分配失败不降级格式。持久资产编辑/Cook/导出和其它平台不据此勾选。）
- [x] 持久 RenderTexture 与图内临时 handle 区分所有权；内部可以导入/暂存，作者不管理内部 RTHandle 式多层包装。
- [ ] 通用离屏 Camera 支持输出、剔除、投影/裁剪、调度和独立 View history；与屏幕相机使用一致 camera 顺序和生命周期。
- [ ] 图依赖决定按需更新、多 pass、双缓冲和读写次序；同一纹理本轮读写冲突明确解决，不靠旧帧兜底。
- [ ] 平面反射作为项目通用能力消费者，正确处理反射变换、剔除/裁剪和递归范围。
- [ ] 公开同 View 的 opaque color/depth、必要法线/运动向量、灯光/阴影、Texture3D 与 buffer；定义 UV、深度重建、分辨率/MSAA resolve 和可读阶段。
- [x] 支持对象集合/submesh 绘制、自定义替代材质与逐 draw 参数；先试现有 queue/pass_tag/override 接口，仅补不足。（2026-09-13：`RendererSelection` 将完整身份/指定 submesh 与捕获参数接入现有 DrawRenderers；精确 submesh 优先，替代材质由集合持有。原生身份/代际/批处理测试与真实11号场景的 GPU 变形、两 submesh 独立颜色、清空/恢复通过；不改变原材质、不复制网格。源材质专属 VS 变形须由替代材质表达，不能等同于已完成的 GPU 网格变形。其它平台及完整游戏效果分别验收。）
- [ ] 支持独立 depth target、depth compare/write、全屏颜色及片段深度输出；受平台限制时明确报告。
- [ ] 动态 RT resize、销毁/in-flight 退休、无 Camera 依赖的计算、UI 取样和导出路径有测试。

验收：监控画面显示到 UI/世界屏幕；反射水面；两 Camera 输出互不污染；渲染目标切换/销毁与资源依赖正确。

2026-09-13 材质持久引用续进：材质仍保存原有纹理 GUID，不引入另一份材质文档；渲染器在资产变更后准备 RenderTexture 并参与现有图读取依赖。导入资源、运行时材质副本和 UI 材质共享同一 GPU owner；临时创建的目标仍是无保存副作用的运行时覆盖。删除只解绑对应资产，保留 GUID；重导入沿用已有 owner 原子换代。Project 拖拽白名单改用统一字段描述，MCP 的 setter-owned 字段不再误用原生读取适配器。真实041Lab已验证 Cube 与屏幕 UIImage 同时显示相机输出、重导入及删除撤销，截图仅人工查看；原场景保留。UIImage.texture 的直接资产槽、反射 demo、Player/多平台等出口仍未完成，当前证据不代表 A09 完整验收。

2026-09-13 Game 相对尺寸续进：公开 `inx.RenderTexture(scale=(0.5, 0.5), ...)`，不同时提供像素宽高；`width/height` 返回真实分配尺寸，`scale` 只读，零碎像素向上取整。比例以 Game 实际渲染分辨率为唯一参考，不随面板缩放、DPI、Scene 或相机绘制顺序变化。只在 Game resize 时处理弱引用目录，全部候选分配成功后一起发布；失效 owner 不被目录保留，固定目标不参与。编辑器面板注册与 Player 激活前准备 Game 输出，因此脚本 start 不依赖 Game 首次可见。原生故障注入证明第二张分配失败时两张旧资源与采样槽均未变化；真实 Vulkan、格式/采样/图回归 3/3、Python 定向 165 项与全量 5963 passed / 11 skipped 通过。可见 041Lab 经 MCP 用真实工具栏改变 Game 分辨率，Camera A 的半尺寸资源 960×540→640×360→1280×720→960×540，Camera B 保持 641×401；两个 UIImage 和 B 中的普通材质取样同一 owner，无逐帧脚本 resize/哈希/隐藏拷贝。Game 隐藏后两台相机仍出图；引擎截帧人工查看，Console 0 warning/0 error，临时内容删除并恢复原 09 场景 clean Edit。与 Unity RTHandle 只比较参考尺寸概念，不暴露第二种作者资源，也不照搬最大历史尺寸的分配策略。文档已中英文同步；正式反射 demo、按需执行、资产编辑/导出和其它平台继续未完成。

2026-09-13 Camera 输出采样合同续进（取代下方早先的 TAA 未通过状态）：原生 `ScriptableRenderContext.output_samples` 在构图前提供固定目标采样数，0 表示由屏幕管线指定。`RenderGraph(output_samples=...)` 和 `set_msaa_samples()` 的有效返回值让 Forward、声明式 DSL、无 RenderStack 默认图及独立自定义管线从一开始构造一致的颜色/深度/motion/normal/resolve 拓扑；不逐 Camera 改写作者参数。RenderStack 按输出采样配置复用图，并把后处理绑定、参数上传版本、编译状态一同保存，显式编辑及反序列化使各配置一起失效。Deferred 仍有原来的单采样限制，对多采样 Camera 目标明确拒绝，不静默改变渲染模式。Windows 构建、原生 2/2 和初步定向 Python 281 项通过；真实 041Lab 双目标 TAA 从此前失败恢复出图，又验证同一 Camera 的 4×→1×→4×、连续 resize、曝光 0.25/2/1、TAA 关闭/重新开启，参考 Camera 不串用状态，最终 Console 0 warning/0 error。完整 Python 回归结果单列执行日志。Unity RenderTexture 也以目标资源决定采样和 resolve，但 Unity 6 URP 不支持 TAA+MSAA 同开，本轮只以其目标所有权和生命周期作比较，不宣称组合功能完全相同；当前测试亦非 TAA 动态重投影画质/性能及全平台验收。严格勾选数不变。

2026-09-13 通用历史帧续进（本节最新状态）：图内 `create_temporal_history()` 使用既有 RenderTexture 所有权生成每 View 独立双缓冲，支持固定尺寸及 View 相对尺寸、raster/copy 写入；首次或失效后的读取明确清零，输出必须先写后读，拒绝本轮同 pass 自读写及修改只读历史槽。历史存储与相机采样抖动分离，只有显式 `set_temporal_jitter()` 才启用抖动，内置 TAA 已显式声明。GPU 真机回读发现了交换 image view 后 Vulkan 已编译 raster 附件仍引用旧 view 的通用问题；现对 color/resolve/depth 建立编译期反向绑定，只刷新受影响附件，不逐帧重建或扫描整图。最后 Windows Release 构建及原生 2/2、Python 182 项回归通过。真实 041Lab 双 Camera→UI 的光迹反馈已验证不同目标尺寸、连续 resize、独立相机跳变和 Game 隐藏后更新；这只是临时功能测试，不是正式漂亮演示场景。**未通过项：内置 TAA 的混合 4×/1× Camera 输出测试与目标全部 1× 的诊断测试均报 motion/depth 采样不匹配。** 当前 RenderStack 仍按作者默认 4×MSAA 共用一份图，原生只改目标采样数，未同步特化 motion/normal/resolve 拓扑；下一步需按实际输出合同构图和缓存，不能靠关闭 MSAA/TAA 或旧帧回退掩盖。按需执行、完整 history/TAA、持久资产编辑/Cook/导出、正式反射水面和其它平台仍未收口，A09 整项不新增勾选。

2026-09-12 反射视图续进（本节最新状态，下列较早记录中的只读 view/未接通消费者仅表示当时状态）：`Camera.view_matrix` 现支持运行时仿射覆盖，`camera_to_world_matrix` 返回实际视图的逆矩阵，`reset_view_matrix()` 恢复当前 Transform 控制；`invert_culling` 是独立的每 Camera 光栅策略，不改共享材质、不使用全局翻面开关。渲染提取、剔除、射线、选中相机视锥、相机局部着色位置与阴影相机范围共用实际视图；翻面进入材质/粒子管线缓存键，并在本次图编译前发布，光源空间 shadow caster 不随观察相机翻面。非法视图仅在公开 setter 边界拒绝，没有逐帧重复校验、旧视图重试或隐藏回退。Windows Release 构建成功，3 项原生测试及 182 项 Python 回归通过。真实可见 041Lab 经 MCP 验证反射、关闭/重新开启翻面、斜近裁剪与 Reset；两个 Camera→UI 对照显示器保持相互独立。额外片段着色器用实际相机 eye.x 显色：反射图青蓝、参考图粉红，Reset 后均为粉红，证明 GPU 也使用各自的真实视图位置。引擎截帧仅人工查看；临时脚本、shader 和对象已清理，原 09 场景恢复 clean Edit，Console 0 warning/0 error。证据见执行记录及 `041-reflection-view-eye-final.log`。独立 history/按需与双缓冲调度、持久资源编辑和导出、正式反射水面场景及多平台仍未完成，未据此勾掉完整 A09 条目；总数保持 128/332（38.6%）。Unity 比较采用自定义 worldToCameraMatrix/Reset 和反射翻面语义，但保留引擎 +Z/[0,1] 坐标及每 Camera 状态，不引入并列资源类型或全局开关。

2026-09-12 投影/裁剪续进：Camera 新增运行时 `projection_matrix`、`reset_projection_matrix()`、`calculate_oblique_matrix()` 与只读 `view_matrix`，沿用唯一原生投影给渲染、剔除、屏幕射线和选中相机视锥；自定义矩阵不改写作者 FOV/near/far，Reset 后恢复这些字段控制。射线从近裁剪面发出，逆投影按投影变化缓存，正交、偏轴和斜近裁剪不再另猜 FOV。坐标明确为引擎现有左手 +Z、深度 [0,1]、屏幕左上原点，不直接照搬 Unity 的 CPU 投影矩阵。原生 2/2、扩大 Python 383 passed / 1 skipped、Camera 集成 5 项通过；真实 041Lab MCP 的两个独立 Camera→UI 显示器验证透视斜裁剪、偏轴、正交斜裁剪及 Reset，参考相机保持不变，最终 Console 0 warning/0 error，临时内容已移除且原场景 clean。选中相机视锥的矩阵几何有单测，但尚未单独完成其自定义投影的真实编辑器操作验收。自定义 view、反射翻转剔除、history、完整反射 demo 和多平台仍未完成，A09 不提前勾选，总数仍为 128/332（38.6%）。

2026-09-12 同图材质/UI 依赖续进：实际 draw 的 RenderTexture 读取现区分图内先行生产与跨相机输入，并在对应 pass 建立时取得当前附件版本；不再于建图起点固定 v0。写入后同步发布同一图资源的各个别名，单采样 color/resolve 仍是同一物理附件。真实 041Lab 经 MCP 验证“渐变生产 → 材质第一次读取 → 同纹理改写 → 材质第二次读取与 UI 读取”，1×/4×及连续 resize 均呈现对应版本；故意把读取置于生产之前会拒绝调度，恢复顺序后不重启即可出图。跨相机 UI、替换资源及缺少生产者的回归也通过。新增真实 Vulkan 前后两次采样回读，六组尺寸/格式/采样和 in-flight 退休通过，相关原生测试 3/3、Python 365 passed / 1 skipped。没有添加旧帧、隐藏拷贝或重试兜底；按需更新、双缓冲/history、持久资产/导出、反射 demo 与多平台仍未完成，因此整项勾选及总进度 128/332（38.6%）保持不变。

2026-09-12 RenderTexture 原生底座回归：持久资源已统一持有颜色、可选深度、MSAA 及可采样 resolve，resize 按完整代际发布，图内借用和 GPU 退休沿用 RHI 所有权。相同代际重复导入返回同一附件的最新图版本，不创建漏掉依赖的独立别名。真实 Vulkan 测试覆盖六组尺寸、UNORM/HDR、1×/4×MSAA、清屏→resolve→compute shader 采样→逐像素回读，以及录制后 resize/销毁、完成序号前不释放旧分配；4 项相关原生测试、73 项 Python RenderGraph 测试通过。公共 Camera/材质/UI 引用、跨 View 调度与导出尚未接通，因此 A09 暂不勾选；这不是监控/反射演示的完整验收。

2026-09-12 公开资源首段：新增 `inx.RenderTexture(width, height, format=..., depth_format=..., samples=..., filter=..., storage=...)` 与 `resize()`，复用引擎当前设备及 RHI 代际所有权，不虚构磁盘 GUID，不暴露 Vulkan 句柄。真实 041Lab 经 MCP 创建临时脚本/组件，在 Play 中创建 1×/2×/4×MSAA HDR 资源并 resize，属性、代际及非法格式拒绝均通过，Console 0 warning/0 error；测试对象和脚本已移除，原场景恢复 clean。86 项 Python 回归通过。该公开接口当前只覆盖固定尺寸的资源分配/生命周期，尚不能据此宣称 Camera、材质、UI 消费者、相对尺寸自动联动或持久资产编辑完成；A09 勾选保持不变。

2026-09-12 持久资源接入渲染图：`graph.import_texture(name, target)` 返回既有图内 handle，描述持有 RenderTexture owner，原生图借用具体代际，重复导入同一资源共享身份。Resize 在下一次图执行前更新尺寸并重建附件/取样绑定；旧代际仍按提交完成序号退休。当前公共入口是单采样颜色、同图每次执行首先清屏写入，不支持在尚无跨 View 调度时读取其它图或旧帧内容；MSAA/depth 的原生资源能力不等于 Python 图入口已经全部贯通。真实 041Lab 经 MCP 创建测试管线和 shader，运行中将 37×23→53×29→113×67→37×23，左半屏采样清屏颜色，右半屏由 shader 的 `textureSize` 编码颜色，引擎截帧确认取样绑定跟随真实尺寸而非旧资源；0 warning/0 error，测试资产和对象已可撤销地删除，原场景 clean。89 项 Python、3 项原生回归通过；删除无调用的字典备用构图路径，统一使用原生 schema。此处只关闭“持久 owner 与临时 handle 分离”一项，Camera/材质/UI 消费、跨图依赖、相对尺寸与完整导出仍未验收。

2026-09-12 MSAA/depth 与编辑器目标统一：公共图入口现已支持 `graph.import_texture(..., attachment="color"/"depth"/"resolve")`，单采样 resolve 与 color 共用身份，多采样颜色通过显式 resolve 取样，单采样深度通过 `sampled_depth=True` 声明取样用途。固定尺寸全屏 pass 使用资源自身尺寸，depth-only pass 不再隐式附加屏幕颜色，Camera 清屏配置也不再覆盖独立持久目标的清屏；尚未实现的多采样深度取样和跨图历史读取仍不开放。Scene/Game 原生目标改用同一 RHI 代际所有权，移除重复的分配/退休代码，相关两文件净减少 209 行；初始化布局合并为一次提交。真实 041Lab 在 4×MSAA 下验证渐变全屏绘制、resolve、独立深度取样及四次尺寸切换；把主 Camera 背景改为红色后，独立目标的蓝色条仍保持正确。Scene 图标及选中描边另行截帧检查通过，临时对象/资产移除后原场景恢复 clean Edit，Console 0 warning/0 error。联合 Python 回归 185 项、相关原生回归 3 项通过。测试同时修正 MCP 原生字段查询与写入的命名/枚举投影，使 Camera/Light/AudioSource 按公开 schema 写入并支持 Undo/Redo；静音仍走字段 setter，不重载音轨。Camera/材质/UI 消费者、跨 View 调度和完整监控/反射验收仍未完成，本段不新增完整勾选项，合计保持 127/332（38.3%）。

2026-09-12 离屏 Camera 前置解耦：排查确认多 Camera 仍共用屏幕输出目标，不能只增加 `target_texture` 属性就算完成。先移除 SceneRenderGraph 材质 callback 对全局颜色格式、深度格式和 MSAA 的依赖，按当前 View 与具体 pass 附件生成管线配置；天空盒同样携带显式配置。当前 View 的投影尺寸来自该渲染图的输出尺寸；callback 缓存识别颜色/深度格式变化，不再仅比较采样数，稳态检查仅比较固定字段。原生矩阵覆盖不同 View、格式、采样、MRT 顺序、只读/取样深度及 depth-only。真实 041Lab 将同一批物体和程序化天空盒分别画入 HDR/4×MSAA/D32 与 UNORM/1×/D24S8 两个持久目标，再并排采样；321×181→533×299→113×67→321×181 动态尺寸序列通过，控制台 0 warning/0 error。144 项 Python 和 3 项原生回归通过。该证据是同一 Camera 的多目标 pass，不是两台离屏 Camera，也不代表 Camera 输出绑定、跨 View 依赖调度或 RawImage 引用完成；A09 清单保持未收口。

2026-09-12 离屏 Camera 输出绑定：`Camera.target_texture` 现持有同一个公开 RenderTexture owner，输出尺寸、格式、深度和 MSAA 来自资源自身；运行期引用不伪造 GUID，不进入场景文档。原生 View 直接借用资源代际，没有第二张中转纹理；resize/同尺寸替换刷新 View 与 history，解绑恢复屏幕输出。显示编码前的图边界在最终拓扑生成时解析，离屏输出保留线性颜色、不附带 Screen UI；相机截图仅在 PNG 编码时转换为显示颜色，不修改目标资源。MCP 新增 `source="camera"` 与稳定 Camera component ID，可直接截取目标纹理，不转用桌面截图。真实可见 041Lab 在原 09 场景临时添加两台不同视角相机，分别输出 HDR/4×MSAA/D32 横图和 UNORM/1×/D24S8 竖图；尺寸 321×181→533×299、同尺寸 owner 替换、113×67、解绑/重绑、返回 321×181，全部出图且两路互不串用。切换到 Scene 隐藏 Game 后两路继续渲染，Scene 图标正常；Console 0 warning/0 error，临时对象/脚本已通过 MCP 删除，场景恢复 clean Edit。原生相关回归 3/3 通过，公开接口/图/Camera/MCP 首组 Python 回归 135 项通过。此处仍不代表世界 UI 离屏绘制、材质/UI 消费、跨 View producer/consumer 调度、相对尺寸、持久资产或监控/反射 demo 完成，A09 全项和严格进度保持不变。

2026-09-12 跨 View 显式依赖首段：Camera/RenderGraph 声明的持久纹理读写生成统一的 View 执行顺序，直接录制和分批提交共用；共享输出维持 Camera depth 顺序，无依赖的 View 不强行串成一条依赖链。相机 B → 相机 A → Scene/Game 在相机 depth 与依赖相反时仍能正确出图；颜色、MSAA resolve 和单采样深度都由当前帧生产者提供。真实编辑器经 MCP 验证 resize、同尺寸替换、隐藏 Game、停用生产者后明确报错以及恢复生产者后恢复出图；无有效调度时 Camera 截图失败，不把旧内容标成新帧。纯原生测试另覆盖环、同目标反馈、缺失生产者及共享输出排序。最新 3 项原生和 166 项 Python 回归通过，临时测试内容可撤销地移除，原 09 场景 clean Edit，最终 Console 0 warning/0 error。这里尚未完成按需执行、双缓冲/history、材质/UI 隐式读取、世界 UI 离屏和导出；真实 Camera 首次使用与共享输出保留内容的 Vulkan validation 矩阵仍需补齐，不能用此次 Release 出图代替这项证明。A09 完整条目继续不勾选，总数保持 127/332（38.3%）。

2026-09-12 材质消费首段：`material.set_texture(name, render_texture)` 直接绑定持久 GPU 资源；不生成 GUID、不改变原材质的资产引用、不自动写盘。原生材质、clone 与描述符持有资源；resize 发布新的采样代际，现有描述符按既有退休机制替换。图在当前 Camera 的剔除列表提交后收集实际材质绑定，给绘制 pass 声明读取并加入跨 View 调度，因此相机 depth 与生产依赖相反时仍先完成生产者。普通无运行时纹理的绘制不会重复参与各 pass 的材质读取扫描。真实 041Lab 通过 MCP 在 09 场景临时放置世界监视器，以内置 Unlit 材质显示另一相机的画面；HDR、1×/4×MSAA、连续 resize、同尺寸 owner 替换、隐藏 Game 后继续更新、生产者失效拒绝及恢复均通过，截帧人工查看。校验层确实加载，日志无 Vulkan VUID；明确触发的缺失生产者错误已记录，恢复和清理后 Console 0 warning/0 error。167 项 Python 回归及重建后的 3 项原生回归通过。UI 绑定、同图内部生产/材质读取的完整排序、参数覆盖/各管线组合、按需/history、资源编辑与导出仍待覆盖，A09 完整条目不提前勾选，严格进度仍为 127/332（38.3%）。

2026-09-12 Camera → UI 闭环：`UIImage.texture = target` 是运行时覆盖，`None` 恢复作者的材质/texture_path；不伪造磁盘路径或 GUID，不读回再上传。UI 使用同一 RHI sampled-color/resolve 代际与现有描述符退休，实际 draw command 的输入参与既有跨 View 调度；仅最后一台屏幕相机合成 screen UI，所有相机都按自身 layer mask/depth 绘制 world UI。离屏世界 UI 管线按真实 HDR/UNORM、MSAA、深度/模板附件匹配，Overlay 对线性相机图像只做一次显示编码。真实可见 041Lab 经 MCP 验证两相机逆序依赖、世界/屏幕监视器、resize、owner 替换、Game 隐藏、仅屏幕 UI 的缓存失效；生产者停用时明确拒绝，恢复后正常。顺带修复被相机隐藏的世界 UI 仍响应鼠标，以及自动尺寸世界文本的绘制、输入、Rect/Gizmo 使用旧作者尺寸。扩大 UI/材质/Camera/RenderGraph 回归 365 passed、1 skipped，原生资源/调度/真实 Vulkan 测试 3/3。跨平台、同图内部生产/材质消费、history/按需、持久资源编辑/导出、反射水面与正式监视器演示仍未完成。

## 14. A10：Runner Long 项目视觉与已有 TAA 收口

### A10.1 原效果按真实实现迁移

依据 PuzzleScreenSpaceOutlineFeature、Registry、Mask/Composite shader：

- [x] 项目登记 Renderer 集合及 invalid/hatched/depth_tested 状态；按逻辑 owner 分组，同一 Rigidbody 下零件共享分组 ID。
- [x] 在不透明阶段后绘制对象数据图，独立 depth 保留选中集合内部遮挡；不能直接用被墙覆盖的普通 picking 图替代。
- [x] 原图存分组 ID、eye depth、非法状态和标记；迁移编码按精度/格式边界选择，不能照抄半精度 ID 而忽略规模限制。
- [x] 按对象边界生成红白描边，分别支持主动斜线、被遮挡斜线和显式遵循场景遮挡。
- [x] 保留邻域深度抑制自身/细草噪声的效果目标，参数与 bias 在项目中管理；合成能输出对应深度。
- [x] 默认描边在后处理前，特殊世界文字可在较晚阶段绘制并使用相应深度策略；不改变普通世界 UI 默认三维规则。
- [x] 对完全遮挡、零件交叠、两个选中对象、透明边缘、旋转/运动和不同投影比较画面。

### A10.2 其它效果与历史

2026-09-13 历史验收：`Camera.reset_history()` 通过运行时 revision 通知该相机的各 View，复用现有历史清空、previous VP 和 jitter 生命周期，不修改相机姿态或影响其它相机。真实几何锁存消费者验证18组连续／重置／反射状态及 Scene/Game 切换；内置 TAA 验证1×／4× RenderTexture、独立相机、尺寸／引用替换、曝光和启停。修复这条即时深度解析链路的 Vulkan 队列使用，不等于整个异步队列计划已完成。最终真实编辑器完整退出日志无验证错误；证据见 `041-depth-boundary-taa.log`、`041-depth-boundary-history.log` 和执行日志。

2026-09-13 选中裁剪表面验收：新增独立 `13_CutoutBodies`，保持12号场景；两个文件材质使用不同条纹密度，表面与对象遮罩显式共享覆盖参数及背面剔除规则，不复制任意项目 shader 的语义。真实可见编辑器完成36组投影／材质／遮挡／深度组合、8组状态旗标、玻璃球 GPU 变形、运动与背面剔除、取消选择、拆分 owner、删除后 Stop／重开。人工查看引擎帧确认条纹孔洞、物体轮廓及遮挡一致；普通实心材质不会继承旧裁剪参数，参数切换不重建图。结合11、12号场景已有零件交叠与多个 owner 证据，A10.1最后一项现勾选。新场景最初 Quad 朝向反了，已修正项目旋转，未关闭两面的剔除来掩盖错误。执行细节见 `041-execution-log.md`。

2026-09-13 细遮挡增补：`Coverage Fence` 默认禁用，启用后可配合 `OutlineThinFence` / `OutlineWideFence` 文件材质做窄／宽裁剪遮挡对照；`OutlineGlass` 是可赋给现有球体的透明文件材质。RenderStack 的 Outline Gallery 参数直接显示绝对／相对 bias 和像素半径，半径0对照逐像素判断。半径只作用于自动遮挡斜线，显式 depth_tested 仍采样当前像素；RGBA32F 分类数据不滤波。32组对照、800×450第二相机及6帧运动采样均经真实引擎捕获，人工观察未做图像像素分析。细线噪点减少，宽遮挡斜线保留，玻璃轮廓随 GPU 几何变形；不把这套确定性栅栏样本冒充所有草材质和任意透明裁剪效果已完成。原生渲染本轮未更改；引擎仅修复通用 InspectorSerializedTarget 的字段语义标识，复用所属 Component、字段声明、输入与 Undo，不为描边增加原生特例。完整原生83项为前一轮证据，本轮完整 Python6136/13skipped及真实 Vulkan 验证层记录见执行日志。

2026-09-13 边界：12号场景是项目视觉能力消费者，未开始一次性修改 Runner Long 原项目。邻域深度和片段深度代码已实现，但细遮挡物和透明/alpha-clip 边缘尚未完成组合验收，A10.1 相应2项仍保留未勾选。公共 `draw_world_ui(layer_mask=...)` 与 UI 帧尾 helper 已支持 Pass 层筛选，和 Camera 掩码取交集，纹理依赖同步筛选；原 UI 渲染器及正常深度规则不变。12号场景已目视验收普通文字经过项目调色、晚绘制文字保留颜色、相机排除文字层、两种投影及普通/原场景深度下的完全墙体遮挡，Console0错误。新建的持久 `OutlineCapture.rendertexture` 经真实 Project 右键创建、尺寸编辑、拖给 Camera、保存重开、Edit/Play相机捕获800×450；默认新建资源现带 D32 深度，底层纯颜色描述默认值不改。演示 `CommitGrade` 曾错误复制到 Camera Target，已用正常全屏绘制修正，未放宽引擎合同。此前240帧整帧平均2.58ms/P95 4.43ms仅属于加文字前的12号场景，不能套用到新增相机/阶段，也不代替 GPU timestamp、Unity同任务对照或刚软体300FPS验收。

- [ ] 水面折射/深度吸收/反射、草、项目 Bokeh、体积大气及绘画效果由项目实现；引擎补 A04/A08/A09 公共能力，不强制内建特效。
- [x] 对齐 TemporalAAEffect 和 create_temporal_history 的实际 API/stub；已有历史机制复用。
- [x] 为小范围 teleport 等提供 View 级显式 history reset；自动 camera cut、投影/分辨率、Play、Scene/Game View 与反射相机分别复验。
- [ ] 动态 mesh motion、透明、描边、DOF、世界文字顺序形成项目管线合同；不依赖偶然 pass 注册顺序。
- [ ] 现有粒子反馈通过项目重新制作/参数映射迁移；不承诺 Unity VFX Graph 自动转换。
- [ ] 性能使用已有观测加 A04 的必要 marker，区分 CPU/GPU/同步成本，记录稳定和首次使用帧。

验收：参考关卡效果可通过公共 Pipeline 表达，无专属引擎“Runner Long 描边组件”。

## 15. A11：可上线桌面游戏的音频基础

首轮目标是 Windows/Linux 立体声设备上的正式游戏；“多声音混合”不冒充 5.1/7.1 或 HRTF。
Android/Web 公共音频样本仍需验证；特殊输出能力在矩阵中明确。

- [x] 2D 音乐/音效保留原始立体声；修正当前无条件 L/R→mono；3D 点声源/立体声空间化有明确策略与 2D/3D blend。
- [x] 距离衰减、方向、最小/最大范围及 listener 选择一致；多场景遵守单世界规则。
- [x] 真实 Bus 路由与必要层级：Master/Music/SFX/Ambience/UI，源/空间/分组/master 增益只应用一次。
- [x] 分组音量、mute、渐变不改变应保留的播放位置；未知分组在配置边界报告，不每回调字符串查找。
- [ ] 有界真实 voice + 虚拟 voice，按优先级/可听性调度；不可听循环维护逻辑游标，恢复不中断时间语义。
- [ ] 并发和池化 one-shot 有明确淘汰/拒绝策略，关键反馈不被低优先级环境声挤掉；限额不是异常或静默崩溃。
- [ ] 短音效共享解码数据，长音乐流式读/解码，预留播放缓冲；不能每个 voice 复制整首音乐。
- [ ] 循环、seek、暂停恢复、淡入淡出/过渡使用可靠播放时钟，尽量不依赖主线程帧率。
- [x] 纠正回调临时分配/不可控互斥等待，参数通过明确的实时安全更新通路进入；不重造无需求的万能音频图。
- [ ] 输出峰值/削波与基础增益平滑可控，处理设备切换、失焦策略、场景卸载和停止。
- [ ] 为必要 DSP 处理保留明确位置；混响/滤波/snapshot 仅按游戏实用需求增补，5.1/7.1、HRTF、完整 Mixer 编辑器不默认为首批交付。
- [ ] 长时间压力测试包含距离声源进出、大量 one-shot、音乐、暂停/恢复、切场景、换设备；测音频 underrun、CPU、内存并实际听音。
- [ ] 基于 A00 实际声源规模冻结 voice/内存/延迟预算，不拍脑袋宣称“无限声源”。

2026-09-10 首个音频纵向切片：`AudioSource.spatial_blend` 已成为 C++/Python/Inspector/序列化统一字段。新建源默认 `0`，保持原始左右声道；`1` 将双声道合成等功率点声源并应用距离衰减/声像，中间值连续混合。旧场景缺少该字段时唯一迁移为此前的全 3D 行为 `1`，重新保存后写入当前格式。AudioClip 预览固定走非空间 2D。实时回调不再逐次构造动态数组或等待参数互斥锁，使用固定栈分块和原子参数快照；声道、声像、距离与 blend 数学有独立原生回归。固定 `Master/Music/SFX/Ambience/UI` Bus 已接入脚本和 Inspector，未知 Bus 在配置边界拒绝，source/track/bus/spatial/master 增益各应用一次。同一 AudioClip/输出采样率现在只准备一份不可变 F32 立体声播放数据，多个 voice 共享所有权；Unload 只切断后续获取，活动 voice 安全持有旧代际。实时回调以固定约 5ms 的采样级斜坡趋近 gain/spatial/pan/blend 目标，不在回调中分配或加锁。voice 虚拟化、长音乐流式解码及真实听音压力矩阵尚未完成。

2026-09-11 Listener 确定性补齐：空间距离衰减移入唯一的 `AudioMixer` 数学实现并覆盖 near/mid/far 与退化范围；Listener 继续是整个 World 唯一活动者，新注册 Listener 不替换当前活动者，活动者禁用/销毁时按稳定 GameObject ID 提升备用者，不再依赖 `unordered_set` 的偶然顺序。公开只增加只读活动 Listener 身份供工具和验收观察。音频集中 Python 回归 `8 passed`，Windows Release CTest `76/76`。A02 Additive Scene 尚未落地，故“多场景遵守单世界规则”仍缺真实多场景证据，本项不提前勾选。

2026-09-11 A11 第二段收口：A02 Additive Scene 已完成，同一 World 中跨 active/additive Scene 的 Listener 选择回归成立，补齐上段保留的多场景证据。`AudioEngine` 新增固定五 Bus 的帧时间渐变、取消和状态查询；mute 只改变最终 gain，渐变和直接音量写入均不 seek、暂停或重建 voice，直接写入会确定性取消现有 envelope。未知 Bus 继续只在 setter/反序列化等配置边界拒绝，实时音频回调只读取已发布的数值状态。图形引擎真实 `tick` 回归覆盖首帧零增量边界、渐变插值、mute、取消、完成和直接写入，集中 Python 音频回归 `10 passed`；原生 `infernux.audio_mixer` 与 `infernux.audio_clip` `2/2 passed`。长音乐流式播放、虚拟 voice 和基于音频播放时钟的过渡仍属于后续未勾选项。

2026-09-12 A11 播放游标与暂停修复：公开 `AudioSource.get_track_time/set_track_time`，读取音频回调的源混音游标、按 clip 秒 seek，保持暂停状态并清除旧队列；此游标不是声卡实际播放时钟，硬件缓冲可有延迟。单轨暂停改为解绑该 stream，不再暂停整个共享 SDL 设备；新建/恢复 voice 不会打破全局暂停。修复 Disable 暂停标记被 `Pause()` 清除导致 Enable 无法恢复，以及尾音尚在队列便被销毁的问题。新增真实 SDL dummy 回调和确定性 PCM 取样回归，覆盖双轨独立推进、全局暂停、Disable/Enable、seek/pitch/loop/EOF 尾音与立体声；原生 `infernux.audio_playback` 通过，Python 音频定向 `9 passed`。长音乐流式、虚拟 voice、播放时钟驱动的 Bus 过渡仍未完成，原整项不勾选。

2026-09-12 测试进程退出修复：真实终端 Ctrl+C 原先会被 native 帧回调作为普通脚本异常吞掉；另有主循环异常绕过 `Engine.exit()`，残留非 daemon 资源观察线程。现在主线程 SIGINT 仅请求帧结束后退出，再抛出中断；正常退出、启动失败和异常退出统一清理插件/MCP、资源观察者和 native engine，清理后恢复 signal/GC 与项目锁。编辑器去掉 `os._exit(0)`，引擎清理去掉 15 秒强杀 watchdog，避免用进程终止掩盖生命周期错误。Windows 真实可见 041Lab 已验证正常窗口关闭和终端 Ctrl+C 后进程及 9713 监听消失；退出与资源协调集中回归 `79 passed`。Linux 尚未在本轮验证，不据此声明跨平台收口。

2026-09-12 A11 音频设备时钟：移除由 `AudioEngine.Update(deltaTime)` 推进 Bus fade 的旧路径；Master/Music/SFX/Ambience/UI 的自动化现在依据 SDL 实际输出块时间推进，没有游戏帧、没有活动 voice 时也能推进，`PauseAll` 冻结时钟，mute 不停游标。完整 Bus envelope 通过单生产者/单消费者三缓冲快照进入设备回调，回调不分配、不加应用层互斥锁、不重试；各 voice 使用已发布 Bus gain 和既有采样级平滑，Master 在 SDL 求和/限幅前只应用一次，避免重复增益和限幅后再降 Master 的失真。公开混音 `output_time`、`output_peak` 与 `saturated_sample_count`：最后一项只表示 SDL 限幅后的满幅样本，不冒充限幅前峰值或专业 limiter。关闭设备时保留 voice callback userdata 到所有 stream callback 结束，修复 raw voice 退出悬空风险。原生真实 SDL dummy 设备对实际 PCM 输出验证分组/主增益、停帧淡出、静音继续播放、切换 Bus、全局暂停、饱和计数、销毁和设备重建；两个原生用例各连续通过 10 次，Python 组件/Listener/退出集中回归 `174 passed`。已重建本机原生模块，真实可见 041Lab 经 MCP 完成 Play/Stop。多设备切换、长时间听音、流式解码、虚拟 voice 仍未收口，未据此提高完整音频验收完成度。

2026-09-12 A11 声部调度增量：实际混音限额默认 64、可配置；按 priority（0 最高）、有效音量和稳定开始顺序选择实际声部，其余及不可听声部维护音频时钟上的虚拟游标。虚拟 loop/seek/pitch/手动暂停/全局暂停、晋升后真实 PCM、Bus mute/unmute 均通过 SDL dummy 设备回归；512 个逻辑声部/4 个实际声部的 owner 调度本机单次采样 P50/P95 约 2.6/2.6µs，仅代表调度函数，不代表全部音频线程开销。one-shot 满池拒绝更轻的新音效，否则替换最轻、同音量最旧声部，增加拒绝次数观测而不刷警告。去掉重复 stream 容器与每源逐帧临时 stream 列表。扩展原生播放测试连续 10 次通过，Python 音频/组件/语义/退出集中 214 项通过。新增 MCP 维护的 `09_AudioVoiceField` 首版，素材由 Inspector 的 AudioClip 引用绑定，12 个真实 AudioSource + 移动 Listener，已 Play/Stop 且 Console 0 warning / 0 error；视觉与交互验收仍在进行。尚未证明 Linux/设备切换、长时间听测、长音乐流式和最终内存/延迟预算，因此不据此勾掉整项跨平台验收。

2026-09-12 A13/MCP 帧时机修复：真实场景搭建暴露脚本资产删除在 pre-GUI 阶段执行、与活动 native lifecycle frame 冲突。排队作者命令现统一在 pre-scene 的 owner 事务内执行，先于脚本 reload 和 native begin-frame；不放宽 dispatch 安全点断言，不为单个删除命令增加兜底。真实 MCP 已成功删除本轮创建的重复测试脚本，帧时序与 MCP/Play/退出定向 69 项通过。普通 Inspector/菜单的全部脚本删除、Undo 和重载矩阵仍需继续验收，A13 整项保持未完成。

验收：不同声道测试素材正确、空间声音可辨、分组生效，正式关卡长时间运行无持续爆音/断音/泄漏。

## 16. A12：光标、Tween 与业务基础

- [x] 分离 visible/relative/confine/warp；Window/Game View 坐标统一，DPI 正确，warp 不产生虚假大 delta。
- [x] 失焦、Stop、viewport 销毁恢复可操作光标；Web 无硬件 warp 能力显式表达，不伪造成功。
- [ ] Tween/Sequence 暂不进入 041 交付；待后续取得 DOTween 源码后单独设计，不在当前引擎主线维护第二套动画服务。
- [ ] UI/业务动画继续使用现有字段和事件语义，暂不依赖 Tween；后续补齐 Unity 风格事件后再评估动画扩展。
- [ ] 卡牌、胜利反馈、机关动画复用统一字段/UI 事件，不把业务算法搬到内核。

2026-09-10 光标首段：公开 `Input.set_cursor_visible/is_cursor_visible` 与 `set_cursor_confined/is_cursor_confined`，并保留既有 lock API。原生端用一个状态应用函数组合用户 lock、编辑器 Scene 捕获、可见性和 confinement；失焦必定释放相对模式/窗口抓取并显示系统指针，聚焦后恢复逻辑请求。visible/confined 不再被 lock 的临时物理状态覆盖。新增 `Input.warp_cursor(x, y)` 使用逻辑窗口坐标，仅在可见非 relative 模式执行；引擎更新逻辑位置并吞掉 SDL 为 warp 产生的匹配 motion delta，Web/语义宿主明确返回不支持。Game View 坐标/DPI、真实边缘拖拽和跨平台窗口行为仍待后续完成，因此本段不勾选整项。

2026-09-11 光标生命周期收口：Game View 隐藏/禁用时直接撤销 gameplay focus 与 cursor lock；Stop 在停止原生 gameplay 的同一帧执行相同释放，不再等待一个可能不会渲染的 Game View；`InxView::Quit()` 在销毁 SDL window 前依次退出 relative、confinement、隐藏状态并清除 InputManager 的窗口句柄。既有输入回归覆盖独立 visible/relative/confine 状态、Game View 逻辑坐标、屏幕 pixel ratio 与 Web/语义宿主拒绝硬件 warp；warp 产生的匹配 motion 在原生唯一入口被吞掉。集中 Python 回归 `97/97`、输入原生回归 `1/1`、Windows Release CTest `77/77`。可见 `Infernux041Lab` 通过 MCP 完成 Play→Stop，随后由正常窗口关闭消息退出，进程返回 `0` 且无终止期错误。

2026-09-11 Tween/Sequence 主链：新增公开 `inx.Tween`、`inx.Sequence`、Ease、Restart/Yoyo/Incremental loop、scaled/unscaled/fixed 更新以及按 owner/target 取消；属性路径统一覆盖 float、向量、颜色、四元数、Screen UI 布局和 opacity，不接受 getter/setter 闭包。绑定方法使用弱引用，捕获生命周期对象的闭包在配置边界拒绝；Disable/Destroy、Stop、下一次 Play 和 Player shutdown 都取消而不触发尚未发生的 callback，正常完成与取消是两个可观察状态。Tween 作为运行时服务向既有 Python 生命周期调度器声明 Update/FixedUpdate 需求，因而不要求项目伪造空 `update()` 组件；无活动 Tween 时不增加 Time/native 跨越。集中回归 `109/109`。真实可见 `Infernux041Lab` 新增 `07_TweenTimeline`：只有 `start()` 的组件成功驱动球体从 `y=0.75` 到 `y≈2.41`，Game render-target 捕获为 1920×1080；Stop 后作者位置恢复为 `y=0.75`，Console 为 0 warning / 0 error。严格进度更新为 `47/246（19.1%）`。

该历史记录已被 041 当前范围 supersede：Tween/Sequence 及 `07_TweenTimeline` 后续移出主线，等待 DOTween 源码与独立计划；当前 041 不分发或加载 Tween 模块。

2026-09-15 Tween 清理后回归：删除 041Lab 的 `07_TweenTimeline` 测试场景与脚本，并从生命周期、Player、Play Mode 和 UISelectable 移除 Tween 运行时依赖。保留 `01/02/03/06/08/09/10/11/12/13/Start` 共 11 个场景，逐场通过 `headless_project_smoke.py --play-frames 30 --allow-unlisted-scene`；每个结果均为 `status=passed`、`runtime_errors=[]`、无缺失对象或组件。批处理终端同时包含日志前缀，汇总解析器无法直接合并 JSON，但逐场 JSON 输出是通过证据。

RenderTexture 作者入口补齐回归：新增 Project File Manager 创建 `.rendertexture` 资产的契约测试，确认默认二进制资源描述、深度目标和重复命名拒绝；`test_render_texture_asset.py` 当前 `39 passed`。这只证明作者创建入口，不提前勾选 A09 的跨 View、Player 和多平台收口项。

Tween 清理后的 Windows Release 原生回归：`ctest --test-dir out/build/windows-msvc-release -C Release --output-on-failure` 全部 `90/90` 通过（含 RenderTexture、Compute/GPU、Mesh publication、场景驻留和 Vulkan 用例），总耗时 117.62 秒。

Tween 清理后的完整 Python 回归再次通过：`python -m pytest python/test -q` 为 `6354 passed, 12 skipped`，耗时 5 分 32 秒；测试中故意触发的负向 C++ 日志均由断言场景消费，不构成运行时失败。

统一 3D 指针事件合同首段：`InxComponent` 现在提供 Unity 风格的 `OnMouseEnter/Over/Exit/Down/Drag/Up/UpAsButton` 可覆写钩子，不新增 `WorldHover` 或其它组件继承层；生命周期与 UI 定向回归 `165 passed, 1 skipped`。本次只冻结公共 Python 表面，Collider 命中分发、按下对象捕获和场景/Player 实际交互仍保持未完成，不能提前勾选 A06/A15。

验收：拖动物体光标不跳，停止后可操作编辑器；取消 Tween 不执行未发生的后续 callback。

## 17. A13：作者工具与资产操作

- [ ] 菜单/命令、快捷键、启用条件、Scene 创建保存、Prefab 保存实例化和现有 Apply/Revert 通过稳定公共入口操作。
- [ ] 核对项目所需嵌套/实例引用/override；复用已存在语义，缺失的必要编辑行为补齐，不冒充完整 Unity Editor SDK。
- [ ] 游戏创建关卡/机关/数据资产的工具使用 A01 操作与字段描述符，不手写 Library/Scene 内部格式。
- [ ] 注册项按插件/项目 owner 生命周期管理，重载不重复，禁用/关项目不残留。
- [ ] 中文命令、Rect/Handle 编辑、批量修改、一次 Undo 与 Prefab 引用保持一致。

验收：策划通过菜单创建可保存机关/关卡，Undo/Redo、重开和导出后值与引用一致。

## 18. A14：插件分类、交付与 Steam 云存档

### A14.1 官方插件分类与旧计算插件退役

- [ ] 复用已有 category/targets；“官方”是来源认证，分类是用途，已安装/可更新是状态，三者独立。
- [ ] 官方分类使用稳定内部键与中英文显示：平台构建、计算与仿真、编辑器与工具、在线服务；搜索/筛选组合可用，不按名称猜分类。
- [ ] Taichi 不再列入官方插件，计算与仿真分类仍供其它插件使用；MCP 归工具，平台插件归构建，Steam 归在线服务；第三方缺分类仍可在 Other 查看，不被丢弃。
- [ ] 插件列表、详情、下载/未导入、已安装和更新状态均显示一致分类；仅增加必要筛选，不建设新的商店后端。
- [ ] Taichi fork 迁至 external/taichi_for_infernux，编译模块随引擎本体 wheel 分发；删除旧 catalog、项目依赖、安装说明及插件 release 流程，编译/平台载荷由构建直接产出，不能靠手工移动文件。
- [ ] 编辑器与 Player 的双 JIT 依赖明确收集并裁剪，移除 AOT 专属载荷；普通用户不通过 CMake 补缺失文件。
- [ ] 保留其它官方插件的安装、分类、文档/配图、更新、禁用/卸载、Python/meta 刷新验收；Taichi 改验收引擎 wheel 安装/升级、旧插件迁移、JIT 打包与运行载荷 ABI。
- [ ] 复用现有官方目录、Cloudflare 优先/GitHub 后备与 Hub 依赖管理规则，不为内置编译器另建插件下载渠道或散落用户磁盘的缓存。

### A14.2 Steam

- [ ] 本期业务只需云存档，不扩大为成就/统计/匹配系统。
- [ ] 稳定应用/用户存档路径、完整写入、版本迁移和本地离线保存由引擎/游戏既有路径完成，不改写 Cook 包。
- [ ] 核对 Auto-Cloud 是否足够当前游戏；需 API 时提供官方 Steam 插件的最小 RemoteStorage 桥接，不耦合核心引擎。
- [ ] 插件形式/官方分类为确定方向；不为只同步文件强制引入完整 Steam SDK，Auto-Cloud 与 API 是明确选定方案而非失败后互相兜底。
- [ ] Steam 回调生命周期、账号隔离、关闭云同步、离线恢复和冲突行为有真实发行配置验证；配置/权限缺失时报告证据缺口。
- [ ] 需要新官方仓库/发行凭据时依正常发布流程执行，不把规划阶段视为已经创建/发布。

验收：普通游戏能本地保存并按选定 Steam 路径同步；插件归类和更新清晰，未装 Steam 不影响不依赖它的项目；计算编译器缺失与 GPU 执行能力按 A05 的最终交付合同区分。

## 19. A15：一次性迁移、集成与发布收口

- [x] 在桌面创建独立 Infernux041Lab，使用真实 MCP 维护场景；新项目不默认安装 Taichi。
- [ ] 重做第一个透明果冻场景：此前 CPU 距离弹簧/体积约束版本被用户否决，不计视觉验收通过。参考稳定 Neo-Hookean 材料及 Small Steps 原论文，以连续体剪切/体积响应、按体积分配质量和低数值耗散实现挤压、回弹、剪切振荡；模拟体与平滑显示表面分别设计。
- [ ] 将测试项目迁移到 inx.buffer + compute.kernel/launch，经引擎共享 Vulkan 执行；使用 wheel 内置编译模块，不要求安装 Taichi 插件。并行写入必须无竞争，核查调度与主机传输成本，不以旧 hpc/NumPy 原型或 CPU 替代宣称新模型完成。
- [ ] 果冻验收通过真实 MCP 操作和连续画面证明落地压缩、弹性恢复、局部扰动传播与稳定静置；数值测试覆盖体积、翻转、能量耗散和不同步长。内部核使用自定义 Gizmo，开关与性能一并验收；截图或高度数值单独均不算通过。
- [ ] 延续同一桌面项目增加后续测试场景；构建后 Player、多平台与官方安装测试另行记录。

### A15.1 随功能落地的可交互演示场景（持续交付）

演示不是最后补拍的宣传图，而是对应公共功能的真实消费者。每个功能阶段完成一批可打开、可操作、可保存的场景；最终与 041 一起交付可运行项目与适用 Player。漂亮的画面、明确的交互和可接受的性能都要成立，不以孤立演示替代 Runner Long 完整关卡验收。

| 场景 | 依赖 | 交互与画面目标 | 必须留下的验证证据 |
|---|---|---|---|
| 01 / 透明果冻实验室 | A03、A04、A05、A07.1 | 落地压缩、局部挤压与回弹；透明平滑表面，可开关内部节点 | 已验证引擎 Sphere 拓扑、文件材质、GPU 驻留表面及自动法线/切线；最新 240 帧完整帧平均 0.695 ms、P95 1.155 ms，但动态刚体耦合仍缺 1/8/32/128 数量矩阵、持续双向接触和 P99 尾延迟收口，不能据此通过 A03.4 |
| 02 / 可交互雪地 | A04、A05、A08、A09 | 用角色或拖动的工具留下连续雪痕，观察压实与光照；支持清除并重新交互 | 已交付首版高度场：公开 `inx.buffer` + 两个 `compute.kernel/launch` 原地更新 65,536 顶点，GPU 结果直供 Mesh 与自动法线；MCP 箭头键移动、快速横扫连续覆盖、边界 clamp、`R` 重置、文件材质、保存重载再 Play 均通过。1920×1080、256² 实测约 1195 FPS / 0.8 ms，单次压实 dispatch 约 0.09 ms，0 warning/0 error。它明确不是体积雪；堆积、Player 导出、分项 timestamp 与连续录像仍待补齐。 |
| 03 / 可交互轮廓选择 | A08、A09、A10 | 悬停/点击物体出现不同描边，遮挡前后仍能区分选择；运行时调色、粗细与样式 | 同一项目自定义管线的对象 ID、深度和材质覆盖；验证透明物体、多对象、分辨率与相机切换，不靠复制模型放大替代需求 |
| 04 / 场景内交互 UI | A01、A06、A07、A09 | 在三维空间放置可旋转的按钮、滑条、文本与控制面板，可操控场景物体 | 深度遮挡、射线坐标、交互焦点、中文字体与布局、不同缩放；Rect Tool 编辑、Undo 与保存重载一致 |
| 05 / 物理游乐场 | A02、A03、A04、A05 | 逐步加入绳/布、关节装置与软硬物体互动，配清楚的控制和反馈 | 每个效果明确实现模型、碰撞及耦合边界；固定步、暂停/单步、重置、多次运行稳定，不能用循环动画冒充模拟 |
| 10 / 平面反射展厅 | A01、A08、A09 | PBR 球体/雕塑、可移动镜面、环绕暂停和裁剪开关；Edit 与 Play 均有反射 | `10_PlanarReflectionGallery` 已经由 MCP 创建并保存；Camera/Renderer/材质均为正式 Inspector 引用。真实 UI 点击、斜面裁剪、物体禁用/启用、两次 Stop/重开和三路引擎截图通过，0 warning/0 error。Windows 1920×1080，两个 240 帧窗口均值 1.336/1.344 ms；此场景的新 Player/跨平台交付尚未完成。 |
| 11 / Renderer 选择研究 | A04、A05、A08、A09 | GPU 波动球体和旋转雕塑，分对象/子网格描边，保留墙后轮廓；选择与调色控件 | `11_SelectionStudies` 由 MCP 创建、保存并重开。文件材质和 Inspector 引用；Play 使用公开 GPU buffer/kernel 与双 submesh Mesh，逐 draw 颜色互不覆盖；MCP 修改选择/颜色/submesh 后连续引擎截帧目视核对，Console0警告/0错误。240帧均值1.128ms/P95 1.838ms；9次选择变更、拓扑构建数不变。最终公共 API 脚本热更新后，3个按钮各点击两次的6次真实输入也通过，Stop/重开保留作者状态。尚不是 Runner Long 完整 ID/eye-depth/invalid/hatch 效果，Player及多平台矩阵仍待完成。 |

- [ ] 用公共场景与组件能力形成统一入口、场景切换、操作提示、重置及参数面板；随着相关 API 成熟替换临时演示 UI，不建第二套私有框架。
- [ ] 每个场景通过 MCP 创建/维护场景对象与资产引用，脚本经过正常资产刷新；保存后重新打开仍可用，不能依赖当前进程里的临时对象。
- [ ] 每个场景先写清目标设备、分辨率、规模、质量与帧时预算，再测首次编译/载入及稳定运行；分别记录 CPU 调度、GPU 等待/执行、NumPy 传输、渲染、内存与固定步次数。预算不足继续优化或明确未通过，不通过降低质量偷偷放行。
- [ ] 使用可分发的原创或许可明确的素材，统一构图、灯光、材质与说明；提供实际交互录像/连续捕获，避免只展示最佳单帧。
- [ ] 依阶段交付：计算基础成熟后先果冻与雪地；管线成熟后轮廓；UI/Rect 成熟后场景 UI；物理与世界功能成熟后扩展游乐场。阶段期间即可由用户人工体验，不等到 041 最后一天。
- [ ] 最终记录每个场景的项目位置、入口、引擎版本、所需插件及版本（不含 Taichi）、操作方式、已测设备、性能与已知限制；验证预编译插件安装和构建后的运行路径。不承诺尚未验证的平台拥有相同的计算功能。

#### 下一批演示的实施边界

- 果冻性能与回弹优先于灯光美化：旧的 343 节点/1296 四面体样本曾测得 GPU 单步中位数 15.68 ms/P95 17.80 ms，不能将其当作当前整帧结果。最新生命周期自动录制边界下，真实 MCP 编辑器 240 帧窗口为完整帧平均 0.854 ms、P50 0.518 ms、P95 1.242 ms、P99 7.395 ms、最大 7.963 ms；该窗口仍不是 A03.4 的验收，因为没有完成连续刚体接触阶段及 1/8/32/128 阶梯矩阵，P99 同步尾延迟也尚未收口。不能偷偷切回 CPU、降低子步或关闭交互达标；后续必须在引擎级资源驻留/提交/同步路径上复测。
- 回弹验收不能只有整体高度。加入局部按压后释放、侧向拨动、重复落地的连续交互记录，观察多个位置的位移/速度、衰减和体积；CPU/GPU 数值一致只是后端一致性证据，不是果冻质感合格的证据。当前空格按高度权重向整个上部施加速度仍是粗糙测试输入，需要真实局部交互。
- 果冻先完成可交付的视觉与交互版本，再把当前技术场景当作回归样本保留。使用实际 Player 检查透明表面的轮廓、光照、压缩回弹和内部节点开关；不能凭计算正确或一张截图宣布美术验收完成。
- 雪地首版采用高度场压痕，不做完整体积雪模拟。工具沿地表连续移动形成压实轨迹；高速移动采用线段覆盖，避免按鼠标事件逐点盖章造成断痕。先实现压实与重置，堆积作为独立后续步骤，不用凭空抬高边缘冒充质量守恒。
- 雪地计算改用 `inx.compute.kernel` / `launch`，公开边界使用 `inx.buffer` 和标量，NumPy 仅作可选交换。工具位置、半径、压实深度可编辑；网格及法线更新使用引擎公共能力。首轮评估 128×128 与 256×256 网格，记录两档实际帧时与传输成本后确定交付规模，不预先宣称达到目标帧率。
- 雪地至少检查静止持续按压不无限下陷、快速横扫不断痕、边界操作不越界、重置恢复、保存重开和 Player 导出。空间 UI 尚未成熟时用现有屏幕 UI 显示提示，后续迁移到场景内控制面板；不为 Demo 引入另一套 UI 系统。
- 轮廓、世界 UI 和物理场景沿上表依赖推进；每个阶段区分“功能验证”“视觉待验收”“可交付”，未完成的效果保持明确标记，不把所有场景拖到最终收口。

### A15.2 游戏迁移与发布

- [ ] Unity→Infernux 只服务 Runner Long，必要脚本留游戏工程；不建设通用 Unity Scene/Prefab/C#/Shader/VFX 自动转换产品。
- [ ] 迁移脚本和规则用公共 API 重写，几何/贴图/配置保持引用，避免重复引入已存在资源；不复制商业素材到引擎仓库。
- [ ] Level02 完整玩法和视觉链路落地，不能各功能 demo 都通过却关卡无法运行。
- [ ] 文本/图片/字体、插件任意依赖文件和 受管 kernel 源码/IR经现有 GUID/Cook/Content.inxpkg；不重新暴露 Assets/Library 项目结构。
- [ ] Windows/Linux 可见编辑器与最终 Player 验证完整关卡；自动测试外辅以 GUI、截图、实际操作和听音证据。
- [ ] Android 验证相应插件载荷/公共基础功能与 内置 GPU JIT；Web 验证公共基础功能及 A05 重新确认的计算样本，不能因裁剪 Taichi CPU 静默取消浏览器计算验收。桌面参考视觉不自动作为移动端等价承诺，差异需明确。
- [ ] 新 native ABI/字段/渲染变更同步平台预编译载荷、wheel、插件版本约束，不只更新 wheel 留旧 Player。
- [ ] 更新官方中英文 API/插件教程、Taichi 示例、分类/安装/构建说明、MCP 使用与 041 功能介绍；不写未实现能力。
- [ ] 通过既有一键流程产生并发布适用制品，核对各平台下载选择；普通导出不编译引擎/Taichi。
- [ ] 最终提交相关 CI 全绿、PR 可合并，实际发布制品与测试版本一致；无机器/账号不能伪造通过。
- [ ] 删除本轮取代的重复实现和一次性引擎侧脚手架，保留有长期价值的回归；不删除用户游戏文件。

## 20. 阶段顺序与技术门槛

| 阶段 | 工作 | 出口 |
|---|---|---|
| P0 | A00；A05 B01/B03/B05 冻结 | 固定资源/执行语义；落实纯编译职责与 wheel 内置迁移，记录 Vulkan/Python 3.13/WebGPU 风险与现状 |
| P1 | A01 与语义附录 S0—S6 | 权威字段/类型、DataAsset、属性事务、基础操作/MCP/Player 目录成立 |
| P2 | A02、A03、A04 | 同 World、固定步/关节、动态几何和资源提交可被项目使用 |
| P3 | A05 B01—B06 | Buffer/CPU JIT/GPU kernel、内部资源互操作、V8 对照及最终 双 JIT 交付成立 |
| P4 | A06、A07、A08、A09 | 统一 UI/Rect、参数覆盖、通用离屏和管线资源成立 |
| P5 | A10—A14 | 项目效果、音频、业务反馈、作者工具、插件分类与存档成立 |
| P6 | A15、A16 | 完整关卡、复合模型导入与统一 Cook 资产、各平台要求、文档、CI、制品交付成立 |

没有依赖的实现可按代码归属安排，但禁止绕过 A01 再造临时 schema。
WebGPU 计算路线须确认；Taichi 的 wheel 内置已定，纯编译 ABI 和迁移依赖须在实施前冻结；失败先处理技术阻塞/请求范围决定，不拖到最终导出才发现。
本计划不在范围尚有技术风险时给出虚假精确总工期；后续按阶段实证估算。

## 21. 完成定义与 042 交接

041 只有在纳入范围的能力、游戏消费者与最终分发制品一起成立时完成；不以源码 API 存在、测试数量或勾选百分比代替结果。

交接必须包含：

- 权威类型/字段/操作/Player 目录合同、版本、native/Python/DataAsset/插件样本和已删除旧路径。
- UI 布局/世界坐标/字体度量、通用 Rect/Handle 与 Undo 的同一编辑通路。
- 多场景、固定步、资源任务同步、动态 Mesh 与 View/history 的所有权和时间边界。
- inx.buffer/CPU jit.compile/GPU kernel+launch 合同、V8 对照与资源互操作；Taichi 纯编译输入输出、wheel 内置及真实后端/平台矩阵、双 JIT、Web 验证和精确载荷。
- Runner Long 关卡结果、音频听测、性能/泄漏、平台限制和本地/云存档结果。
- `.blend`/FBX/GLTF 编辑期源资产、稳定复合模型子资源身份、重导入规则，以及 Player 仅消费统一 Infernux 二进制模型资产的证据。
- 最终 CI、PR、wheel/插件/Hub（如受影响）和发布位置的对应版本证据。

042 不重做以上基础，消费同一目录实现新 Python 模块的 prepare/inspect/publish/revise/retire、实例迁移、持久 runtime overlay、MCP 动态工作流及四端动态 Python。
Taichi kernel 编译与 Python 应用脚本发布是不同能力；没有 GPU 编译能力的 Player 仍可支持 042 普通动态 Python。
通用 Batch World、张量数据面和 Torch 不是 041 的隐含依赖，也不因引入 Taichi 就被宣称完成。

## 22. Taichi 交付与署名收尾（发布必需）

最终交付为引擎本体 wheel 内的裁剪编译模块，不单独分发 Taichi wheel 或 infernux_taichi.inxpkg。恢复实施后将 external/plugins/infernux_taichi 迁至 external/taichi_for_infernux；CMake/CI 直接产出引擎正式载荷并更新分发依赖。CPU/GPU 双 JIT 载荷按 A05/B06 一起打包，无需项目插件安装；AOT 暂不交付。

- [ ] fork 根目录保留上游 Apache-2.0 `LICENSE`，新增 `NOTICE`，说明源自 Taichi、原作者归属、上游地址以及 Infernux 修改；不将上游作品改署为仅由 Infernux 创作。
- [ ] 保留源码中原有版权、作者及许可标识；修改过的上游文件注明修改，后续裁剪不批量抹除原作者标识。
- [ ] 包装构建将 `LICENSE`、`NOTICE` 直接带入最终引擎 wheel/平台 Runtime 分发目录，不依赖仓库外层 README 承担随包告知；按实际载荷收集所分发第三方依赖的许可证与必要 NOTICE。
- [ ] 对最终包含 Taichi 代码的引擎 wheel/平台载荷及导出的 Player 分别核对许可文件是否随被分发的 Taichi/第三方代码保留；安装、AOT 裁剪和再导出不能误删这些文件。
- [ ] 发布验收确认本体 wheel 可安装/更新使用、内置编译载荷完整；必须移除过时插件安装指引，不把生成 wheel 或 native 编译成功当成最终交付完成。

## 23. A16：`.blend` 复合模型源资产导入与运行时格式统一

这是 041 的最后一项功能范围；它扩展现有统一资产主链，不建立只供 Blender 使用的第二套场景、材质或运行时资源系统。

`.blend` 与 FBX/GLTF 同属编辑器侧模型源格式，但它天然包含对象层级、多个 Mesh、材质、贴图和 Blender 场景语义，因此不能按“一个裸 Mesh 文件”导入。项目中的 `.blend` 保留为可重导入源资产；运行时和最终 Player 不认识 Blender 格式，也不携带原始 `.blend/.fbx` 文件。

这里采用明确的三段式主链：源文件只负责编辑与重导入，Importer 产出可观察的 Infernux 复合模型及子资源，Cook 再将这些产物转换为平台无关或平台优化的引擎二进制资产。运行时 API、场景引用和脚本只面对 GUID、稳定子资源 ID 与 Infernux 类型，不泄漏 Blender/FBX/GLTF 的路径和解析器语义。

- [ ] 建立唯一的复合模型导入主链：确定并固定受支持 Blender 版本与一个正式解析/转换后端，不用 Assimp、临时 Python 解析和 Blender CLI 多路失败回退；`.blend`、FBX、GLTF 的导入设置、依赖和错误统一进入现有 GUID/AssetDatabase。若 `.blend` 转换需要 Blender 运行时，必须由 Hub/引擎明确管理兼容版本，不能要求普通用户自行配置命令行环境。
- [ ] 定义统一的 Infernux `ModelAsset`（暂名）及导入描述：根节点、对象/父子层级、Collection 归属、对象名称、局部 Transform、原点/pivot、可复用实例、Mesh 分件、材质槽、骨骼、动画以及可映射的相机/灯光均具有稳定子资源 ID；拖入 Scene 时按源层级实例化，不合并成一个裸 Mesh，也不把每个子对象伪装成互不相关的顶层资产。
- [ ] 冻结坐标与结构规则：Blender 的右手坐标、Z-up、单位缩放、负缩放、镜像、父子变换和骨骼 bind pose 只在导入边界转换一次；普通 Object、Empty、Armature、Collection Instance 的可保留范围明确。导入模型实例允许在场景中增加用户组件和 override，但源层级仍可被识别和重导入。
- [ ] 建立几何与动画转换边界：保留位置、索引、法线、切线、多 UV、顶点色、材质槽、blend shape、基础蒙皮、骨骼层级和 Animation Clip；能稳定烘焙的修改器、约束、曲线/文字、几何节点或模拟结果在导入边界转换，其余给出精确的不支持项，不在 Player 内运行 Blender 语义。
- [ ] 建立 Blender 外观映射：优先权威映射 Principled BSDF 的基础 PBR 参数、颜色空间、贴图、透明/Alpha Clip、法线贴图、发光、双面与能够转换的基础材质表现；打包贴图和外部贴图都成为独立 GUID 依赖，不把任意 Blender 节点图伪装成完整支持。无法映射的节点必须在导入报告指出具体材质和节点，不静默变成错误外观。
- [ ] 明确“部分保留表现”的首批边界：对象可见性、平滑/硬边、材质分配、骨骼动画、基础相机参数和点/方向/聚光灯可转换；Blender 工作区、编辑器灯光、Compositor、任意 Python Driver、流体/布料运行时、复杂节点图不作为 Player 语义。需要保留视觉结果的内容必须在导入时烘焙为 Mesh、纹理或 Animation Clip。
- [ ] 重导入必须保留稳定子资源身份、已有场景引用和用户 override；稳定身份不能只依赖易变化的数组序号。重命名、同层重排、局部几何/材质修改只更新对应导入产物；删除源对象时报告将失效的引用，不通过删除并重建整棵层级破坏对象身份。
- [ ] Cook 将 `ModelAsset`、Mesh、材质、贴图、骨骼和动画序列化为统一二进制 Infernux 资产并写入 `Content.inxpkg`；可按依赖与流式需求拆成多个二进制块，但它们都使用统一引擎格式。Player 只按 GUID/依赖图加载，最终制品不携带 `.blend/.fbx/.gltf` 原文件、Importer 或中间目录，也不暴露源项目结构。
- [ ] 统一运行时加载与编辑器预览：Scene、Prefab/DataAsset、脚本和 MCP 获取的是同一种 `ModelAsset`/子资源引用；编辑器不能偷偷从源文件读取一套、Player 再走另一套。模型实例化、卸载、依赖驻留和资源生命周期复用现有 AssetDatabase/Cook/Content.inxpkg 主链。
- [ ] 建立真实 `.blend` fixture 和回归矩阵：覆盖父子对象、Empty/Collection Instance、多个 Mesh、多材质与内嵌/外部贴图、透明材质、硬边/法线、骨骼动画、相机/灯光、一个可烘焙表现及一个明确不支持的节点；验证拖拽实例化、Rect/Transform 编辑、用户组件/override、重导入、保存重开、Windows/Linux 编辑器和适用 Player 导出。
- [ ] 增加最终包审计：导出目录和 `Content.inxpkg` 清单不得出现源模型扩展名、绝对源路径、Blender 临时文件或可直接还原的项目目录；同时验证模型在无 Blender 安装的目标机可正常加载。该审计用于确认统一资产主链，不宣称不可逆 DRM 或密码学加密。

验收：同一个 `.blend` 在 Blender 和 Infernux 中具有可识别的一致层级、局部姿态、pivot、网格分件、基础材质贴图、骨骼与动画；拖入场景后仍能按原结构编辑和挂组件。修改源文件后稳定重导入，已有场景引用、用户组件与 override 不失效；在未安装 Blender 的目标机上，导出游戏只靠 `Content.inxpkg` 中的统一 Infernux 二进制资产得到一致结果，包体不含原始模型文件。

### 2026-09-15 PC Player 与包体复核补记

- 用户已在实际 PC Player 中确认软体恢复运动，原“球停在空中”的阻断解除；该确认不替代双向耦合数量增长与帧时间矩阵的完整验收。
- 当前 `Infernux041Lab-PC-current` 实测总文件大小为 161,894,947 bytes（154.4 MiB）。`Parallel.inxmod` 为 42,994,368 bytes，保持压缩封装且不在发布目录展开；项目 `Content.inxpkg` 仅 406,976 bytes。没有裸露的 Assets/Library/ProjectSettings 项目目录。
- 主要剩余体积来自公共运行时：OpenBLAS 20,495,360 bytes、引擎原生模块 12,011,008 bytes、字体 10,801,524 bytes、GPU JIT 原生编译器 9,352,704 bytes。继续优化应针对运行时封装与依赖裁剪，不通过移除 JIT 能力、中文字体或有效项目资产制造虚假的体积下降。当前包体优化仍未收口。
- 同项目 `release` Player 复建为 161,846,798 bytes，仅比 development 少约 48 KiB；因此调试配置不是主要来源。release 设计上不带开发控制服务，不能使用开发态自动输入 smoke 脚本作为验收入口。
- 最新 development Player 已重新构建并通过真实窗口 smoke：`01_XPBD_Jelly` 中 `GPUJelly.simulated_seconds=1.30`，`last_gpu_batch_ms=2.258`，球体 Y 位移 `-0.858`，fatal count 为 0，正常退出耗时约 0.20 s。原生 InxPack、PlayerHost、渲染图、计算资源生命周期与 Vulkan 提交回归共 9/9 通过。
- UI 交互边界已调整：Tween/Sequence 不属于 041，旧的实验字段、测试和运行时钩子已移除；DOTween 风格动画另立后续计划。UI 仍保留 ColorTint 等既有指针状态机，待真实 Editor/Game/Player 视觉验收后再冻结 A06/A15。
- Headless 边界修复：041Lab 的 `GPUJelly` 在无图形设备时现在只完成 authored mesh/scene 检查并跳过 GPU buffer 创建，不再刷出 `_gpu` 缺失异常；源码 headless Jelly smoke status=passed。重新构建的图形 Player 仍通过 GPU smoke（`simulated_seconds=1.08`、Y 位移 `-0.982`、`last_gpu_batch_ms=2.199`、fatal count=0）。
- 验收脚本也已固定优先加载当前 checkout 的 `python/` 源码，避免 conda 环境中旧版已安装 wheel 抢先导入；不设置 `PYTHONPATH` 的源码 headless Jelly smoke 仍输出 `status=passed`，并成功加载 RenderStack/GPUJelly/UI 组件。
- RenderTexture 主链定向回归重新执行：`test_render_texture.py`、`test_render_texture_asset.py`、`test_temporal_history_api.py` 与 `test_render_view_samples.py` 共 **96 passed**；该结果确认持久资源、Camera GUID 引用、View history API 与输出采样合同没有被最近的 headless/hover 改动破坏，但不提前宣称 A09 的离屏 Camera、多平台和正式演示已收口。
- 原生全量回归首次暴露并修复场景驻留门禁：共享 `RuntimeChangeJournal` 默认历史由 256 收紧到 32 个批次，保留有限延迟消费者窗口并沿用 history-lost 的 full-resync 语义；041Lab 场景循环中此前每轮累积的 `RuntimeDomainChanges/_CommittedBatch` 不再持续增长。最终 Windows Release CTest **90/90 passed**（86.03s），其中 `scene_residency_soak` 单项也稳定通过；Python 运行时日志回归 **50 passed**。
- 修复后的源码已重新打包为压缩 Windows Player：`041-pc-journal-fix.json` 为 `status=passed`、无 diagnostics、构建耗时 13.60s；真实 `windows_player_smoke` 的 `01_XPBD_Jelly` 运行通过，Jelly Y 位移 `-1.0508`、fatal count=0、正常退出约 0.17s。该证据确认运行时日志修复进入最终 Player，而非仅限测试解释器。
- 验收脚本新增显式 `--allow-unlisted-scene`，只允许 Editor/headless 通过 `SceneFileManager` 打开未列入 Build Settings 的作者场景，默认仍严格遵循 Build Settings。已用它回归 `12_OutlineOwners.scene`（29 个对象）与 `13_CutoutBodies.scene`（30 个对象）；同时修复 `OutlineGallery` 在 headless 下误建 GPU buffer 的问题，13 场景现在无 `ComputeCapabilityError` 生命周期异常并稳定通过。
- 批量 headless 回归覆盖 02/03/06/07/08/09/10/11/12/13 共 10 个 041Lab 场景；首次发现 `09_AudioVoiceField` 的 Listening Floor、Listener Marker 与 12 个发声体仍只保存了 inline mesh + 空材质槽，原生场景提交因此拒绝。已为它们补回现有 `LabFloor`/`GalleryGold`/`ContactCubes` GUID 材质引用，保留统一资产主链；10/10 场景现均 `status=passed`，无场景提交错误。
- 3D Hover/Raycast 主线补充公共 `Physics.raycast_screen(camera, screen_position, viewport_size, ...)`，统一使用 Game View 的 top-left 像素坐标、`Camera.screen_point_to_ray` 与既有物理 Raycast，不创建第二套命中逻辑；Python 物理回归 `47 passed`。该 API 先作为场景 Hover、拾取和交互脚本的共同底座，尚不把普通 3D Hover 视觉事件误报为已完成。
- 批量 headless 日志审计发现 `02_InteractiveSnow` 与 `11_SelectionStudies` 原先虽然报告 `status=passed`，但 GPU 初始化异常被生命周期层吞掉；已在两个项目消费者中按图形能力边界跳过 headless GPU buffer 创建，保留图形 Editor/Player 路径。新增验收脚本现把 ERROR/ASSERT/EXCEPTION 生命周期日志纳入失败条件；修复后两场景均 `status=passed`、`runtime_errors=0`，相关接受测试 `15 passed`。
- 在输入层补齐 `Input.game_viewport_size` 及 Editor/Player 的每帧发布，和既有 `game_mouse_position` 使用相同显示像素合同；`Physics.raycast_screen` 因此可直接用于普通 3D Hover/拾取而不猜测渲染目标尺寸。输入、物理、Game View、Player 控制联合回归 `137 passed`。
- 视口 API 进入真实 Windows Player：`041-pc-hover-raycast.json` 构建 `status=passed`、诊断为空、压缩资源构建耗时 13.23s；同一制品的窗口 Jelly smoke 通过，Y 位移 `-1.0508`、fatal count=0、正常关闭约 0.18s。该结果确认新输入契约没有破坏 Player 的 GPU/物理运行路径。
- Unity 风格的 3D 鼠标事件不单独引入 `WorldHover` 组件；后续应在统一组件事件分发层按命中 Collider 直接调用 `OnMouseEnter`、`OnMouseOver`、`OnMouseExit`、`OnMouseDown`、`OnMouseDrag`、`OnMouseUp` 与 `OnMouseUpAsButton`。该 API 对齐 Unity MonoBehaviour，不增加额外继承层，也不把视觉效果或 Tween 绑定到引擎核心。

### 2026-09-15 Unity 鼠标回调首个实现

- `InxComponent` 已提供 Unity 对齐语义的 Python 风格、零参数 `on_mouse_enter`、`on_mouse_over`、`on_mouse_exit`、`on_mouse_down`、`on_mouse_drag`、`on_mouse_up`、`on_mouse_up_as_button` 回调；它们是普通可覆盖 Python 方法，不新增 `WorldHover` 组件或继承层。命中详情不作为回调参数，保持 Unity MonoBehaviour 语义。
- Editor Game View 与 Player 共用 `MouseEventDispatcher`：每帧复用 `Physics.raycast_screen` 的屏幕射线，按 Collider 命中对象维护 hover/pressed 状态，并按 Unity 顺序调用对应的零参数 Python 风格 `on_mouse_*` 方法。无 UI surface 的场景也会继续走 3D 命中路径。
- 定向回归 `79 passed`；该结果只确认回调顺序、状态迁移和共享查询边界，真实窗口中的视觉反馈、触摸/多指扩展及完整场景验收仍未勾选完成。
- 041Lab 无图形窗口 headless 回归再次覆盖 `03_WorldSpaceUI`、`09_AudioVoiceField`、`11_SelectionStudies`、`12_OutlineOwners`：四场景均 `status=passed`、`runtime_errors=[]`，对象/组件加载分别为 9/19、23/73、16/38、29/65。该批次只证明脚本域、场景生命周期和组件激活没有被鼠标事件分发改坏，不替代可见窗口输入与多平台验收。
- Unity API 边界复核后移除普通 GameObject 的非官方 `OnClick()`：Unity 的 MonoBehaviour 官方回调列表包含 `OnMouseEnter/Over/Exit/Down/Drag/Up/UpAsButton`，不包含 `OnClick`；`Button.onClick` 只属于 UI Button。Python 公共面使用零参数 `on_mouse_enter/over/exit/down/drag/up/up_as_button`，只改变命名风格，不改变 Unity 的触发语义。Screen UI 与 World UI 保留现有 pointer 事件和 `UIButton.on_click`。定向回归 `32 passed`，完整 Python 回归基线 **6356 passed / 12 skipped**。
- UI 输入快照性能首个增量修复：`collect_runtime_ui_input_surfaces` 现在使用 `runtime_canvas_snapshot_token`，Canvas membership/scene epoch 未变化时复用已排序 Canvas 快照，不再每个鼠标帧重复扫描和排序；世界 UI 元素仍按结构版本独立刷新。回归 `test_runtime_screen_ui_submission.py` 为 **24 passed**，并新增测试证明稳定 token 只收集一次。该改动只是静态快照层优化，布局、材质、命中和 GPU 提交的完整性能矩阵仍待完成。
- 世界 UI 输入批处理与官方 API 回归合并复跑：`65 passed`。这覆盖 Canvas-free 元素投影、遮挡、捕获拖拽、Canvas 快照复用和普通 GameObject `OnMouse*` 顺序；没有引入非 Unity 的 `OnClick` 或第二套 WorldHover 类型。
- UI 渲染提交端同步使用同一 `runtime_canvas_snapshot_token`：静态场景的 Screen/Camera Canvas 不再在每个 RenderGraph 提交帧重复排序；场景结构或 Canvas membership revision 变化时才重新收集，世界元素仍由独立结构缓存管理。RenderTexture/UI 输入提交回归 **33 passed**，041Lab 世界 UI headless 继续 `passed`。这只是快照/排序层的优化，尚未宣称完整 UI 性能收口。

## 24. A17：高性能 Raycast 与非凸网格查询

Raycast 是物理世界的正式查询能力，不允许每次调用遍历全部 GameObject、临时拼装形状或重新烘焙 Mesh。主路径复用物理世界已经维护的 broad phase、shape acceleration structure 与稳定对象身份；批量查询共享一次世界快照和调度边界，不通过结果缓存、内容哈希或多套 fallback 掩盖慢路径。

- [x] 定义统一 `Physics.raycast` / `raycast_all` / 批量射线接口，覆盖起点、方向、最大距离、layer mask、trigger 策略以及 closest/all hit；返回稳定对象/Collider 身份、命中距离、世界点、几何法线和必要的 sub-shape/triangle 信息。
- [x] 单次与批量查询直接进入同一原生查询世界；批量射线不逐条跨 Python/C++ 边界，也不为每条射线复制场景状态。为鼠标拾取、世界 UI、AI 感知与大批弹道保留共同的结构化输入/输出。
- [ ] 基础 primitive、compound 与 convex hull 使用物理后端的加速查询；静态/运动学非凸三角网格建立并复用 BVH，支持真实表面命中而不是退化为包围盒或强制凸包。动态非凸网格的更新成本和允许频率需显式定义，不把高频重烘焙伪装成廉价查询。
- [ ] Mesh 导入、程序化 Mesh 与显式 `MeshCollider.recook()` 发布同一不可变碰撞代际；在途查询只引用已发布代际，替换后按资源退休规则释放旧 BVH，不增加逐射线 SHA256、旧数据猜测或静默回退。
- [ ] 查询与固定步并发边界明确：一帧读取一个已发布物理快照；结果不因容器迭代顺序变化。编辑器 Scene Pick、运行时物理 Raycast 与 MCP 观察共享命中语义，但渲染 Object-ID 拾取不冒充物理表面查询。
- [ ] 建立 primitive、compound、凸包、静态非凸 Mesh、背面/边缘、缩放层级、layer/trigger、销毁/recook 的正确性矩阵；与现有碰撞接触结果核对空间和法线约定。
- [ ] 记录 1、100、1,000、10,000 条射线在稀疏/密集场景的 P50/P95、吞吐和主线程时间；同时增加 Collider 数量与三角形数量，证明开销来自加速结构查询而非线性扫描。预算以 Runner Long 实际关卡基线冻结，不能通过减少命中信息或关闭非凸查询达标。
- [ ] 为鼠标/触摸事件增加专门的 Raycast 查询预算矩阵：1 个视口单射线、静止悬停、拖拽捕获、UI 与 3D 重叠、多个相机以及批量指针输入分别测量；验证事件层不会重复执行同一帧相同射线，也不会在 UI 已命中后无条件扫描全部后方物体。若采用 C++ 批量快照或异步查询，必须验证场景修改、销毁、切场景和 Play/Stop 时的代际边界。

验收：Runner Long 的交互拾取、遮挡检查和批量感知可走同一公共 API；复杂静态场景中的非凸 Mesh 能返回真实三角表面命中，批量查询没有逐条 Python 往返，新增 Collider 时性能不会按全场景线性恶化。

当前证据：Windows Release 原生后端已验证单次/all/批量公共接口与复用输出；非凸 Mesh 命中可返回 cooked triangle identity。单个大型静态 Collider 的批量基线为 1/100/1,000/10,000 条射线约 0.004/0.020/0.175/1.736 ms。1.736 ms 明确不可接受，只证明批量调用边界成立；本项移入 041 后续物理性能专项，需要大规模优化查询调度、后端遍历和输出写入，并补齐稀疏/密集场景 P50/P95 与 Collider/三角形增长矩阵后才能冻结预算。当前先推进 A06 等更大的主线，不用宽松门槛提前收口 A17。
