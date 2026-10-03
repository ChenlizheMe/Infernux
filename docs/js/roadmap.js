const GALAXIES = [{"key":"foundation","titleKey":"roadmap.galaxy.foundation.title","branches":[{"status":"live","key":"roadmap.galaxy.foundation.branch.0","en":"DETERMINISTIC CORE","leaves":[{"status":"live","key":"roadmap.galaxy.foundation.leaf.0.0","en":"Scene ownership","zh":"场景所有权"},{"status":"live","key":"roadmap.galaxy.foundation.leaf.0.1","en":"Fixed-step clock","zh":"定步时钟"},{"status":"planned","key":"roadmap.galaxy.foundation.leaf.0.2","en":"Replay contracts","zh":"回放契约"},{"status":"planned","key":"roadmap.galaxy.foundation.leaf.0.3","en":"Runtime invariants","zh":"运行时不变量"}],"zh":"确定性核心"},{"status":"live","key":"roadmap.galaxy.foundation.branch.1","en":"WORLD & SCENE","leaves":[{"status":"live","key":"roadmap.galaxy.foundation.leaf.1.0","en":"Scene graph","zh":"场景图"},{"status":"live","key":"roadmap.galaxy.foundation.leaf.1.1","en":"Prefab authoring","zh":"Prefab 创作"},{"status":"planned","key":"roadmap.galaxy.foundation.leaf.1.2","en":"World snapshots","zh":"世界快照"},{"status":"future","key":"roadmap.galaxy.foundation.leaf.1.3","en":"Streaming worlds","zh":"流式世界"}],"zh":"世界与场景"},{"status":"live","key":"roadmap.galaxy.foundation.branch.2","en":"PLATFORM LAYER","leaves":[{"status":"live","key":"roadmap.galaxy.foundation.leaf.2.0","en":"Windows player","zh":"Windows Player"},{"status":"live","key":"roadmap.galaxy.foundation.leaf.2.1","en":"Linux player","zh":"Linux Player"},{"status":"planned","key":"roadmap.galaxy.foundation.leaf.2.2","en":"Android player","zh":"Android Player"},{"status":"planned","key":"roadmap.galaxy.foundation.leaf.2.3","en":"Capability matrix","zh":"能力矩阵"}],"zh":"平台层"},{"status":"live","key":"roadmap.galaxy.foundation.branch.3","en":"MEMORY & JOBS","leaves":[{"status":"live","key":"roadmap.galaxy.foundation.leaf.3.0","en":"Thread pool","zh":"线程池"},{"status":"live","key":"roadmap.galaxy.foundation.leaf.3.1","en":"Resource lifetime","zh":"资源生命周期"},{"status":"planned","key":"roadmap.galaxy.foundation.leaf.3.2","en":"Arena allocator","zh":"Arena 分配器"},{"status":"future","key":"roadmap.galaxy.foundation.leaf.3.3","en":"Deterministic scheduling","zh":"确定性调度"}],"zh":"内存与任务"}],"titleZh":"基础架构"},{"key":"rendering","titleKey":"roadmap.galaxy.rendering.title","branches":[{"status":"live","key":"roadmap.galaxy.rendering.branch.0","en":"VULKAN PLAYER","leaves":[{"status":"live","key":"roadmap.galaxy.rendering.leaf.0.0","en":"Forward routes","zh":"Forward 路径"},{"status":"live","key":"roadmap.galaxy.rendering.leaf.0.1","en":"Deferred buffers","zh":"Deferred 缓冲"},{"status":"live","key":"roadmap.galaxy.rendering.leaf.0.2","en":"Shadows & lights","zh":"阴影与光照"},{"status":"planned","key":"roadmap.galaxy.rendering.leaf.0.3","en":"GPU captures","zh":"GPU 捕获"}],"zh":"Vulkan Player"},{"status":"live","key":"roadmap.galaxy.rendering.branch.1","en":"WEBGPU PLAYER","leaves":[{"status":"live","key":"roadmap.galaxy.rendering.leaf.1.0","en":"Browser scene path","zh":"浏览器场景路径"},{"status":"live","key":"roadmap.galaxy.rendering.leaf.1.1","en":"Descriptor contracts","zh":"描述符契约"},{"status":"planned","key":"roadmap.galaxy.rendering.leaf.1.2","en":"Visual parity","zh":"视觉一致性"},{"status":"future","key":"roadmap.galaxy.rendering.leaf.1.3","en":"Portable compute","zh":"可移植计算"}],"zh":"WebGPU Player"},{"status":"live","key":"roadmap.galaxy.rendering.branch.2","en":"MATERIALS","leaves":[{"status":"live","key":"roadmap.galaxy.rendering.leaf.2.0","en":"Typed materials","zh":"类型化材质"},{"status":"live","key":"roadmap.galaxy.rendering.leaf.2.1","en":"Shader stages","zh":"Shader 阶段"},{"status":"planned","key":"roadmap.galaxy.rendering.leaf.2.2","en":"Material instances","zh":"材质实例"},{"status":"future","key":"roadmap.galaxy.rendering.leaf.2.3","en":"Runtime variants","zh":"运行时变体"}],"zh":"材质系统"},{"status":"live","key":"roadmap.galaxy.rendering.branch.3","en":"FRAME GRAPH","leaves":[{"status":"live","key":"roadmap.galaxy.rendering.leaf.3.0","en":"Pass scheduling","zh":"Pass 调度"},{"status":"live","key":"roadmap.galaxy.rendering.leaf.3.1","en":"Transient resources","zh":"瞬态资源"},{"status":"planned","key":"roadmap.galaxy.rendering.leaf.3.2","en":"Frame budget","zh":"帧预算"},{"status":"planned","key":"roadmap.galaxy.rendering.leaf.3.3","en":"Render inspection","zh":"渲染检查"}],"zh":"帧图"}],"titleZh":"渲染"},{"key":"animation","titleKey":"roadmap.galaxy.animation.title","branches":[{"status":"live","key":"roadmap.galaxy.animation.branch.0","en":"TIMELINE","leaves":[{"status":"live","key":"roadmap.galaxy.animation.leaf.0.0","en":"Animation clips","zh":"动画片段"},{"status":"live","key":"roadmap.galaxy.animation.leaf.0.1","en":"Curve playback","zh":"曲线播放"},{"status":"planned","key":"roadmap.galaxy.animation.leaf.0.2","en":"Markers","zh":"时间标记"},{"status":"planned","key":"roadmap.galaxy.animation.leaf.0.3","en":"Layer blending","zh":"层混合"}],"zh":"时间线"},{"status":"live","key":"roadmap.galaxy.animation.branch.1","en":"SKELETAL","leaves":[{"status":"live","key":"roadmap.galaxy.animation.leaf.1.0","en":"FBX import","zh":"FBX 导入"},{"status":"live","key":"roadmap.galaxy.animation.leaf.1.1","en":"Skinned meshes","zh":"蒙皮网格"},{"status":"planned","key":"roadmap.galaxy.animation.leaf.1.2","en":"Retargeting","zh":"重定向"},{"status":"future","key":"roadmap.galaxy.animation.leaf.1.3","en":"Pose cache","zh":"姿态缓存"}],"zh":"骨骼动画"},{"status":"live","key":"roadmap.galaxy.animation.branch.2","en":"PARTICLE GRAPH","leaves":[{"status":"live","key":"roadmap.galaxy.animation.leaf.2.0","en":"GPU emitters","zh":"GPU 发射器"},{"status":"live","key":"roadmap.galaxy.animation.leaf.2.1","en":"Init / Update stages","zh":"Init / Update 阶段"},{"status":"planned","key":"roadmap.galaxy.animation.leaf.2.2","en":"Mesh particles","zh":"网格粒子"},{"status":"future","key":"roadmap.galaxy.animation.leaf.2.3","en":"Simulation fields","zh":"模拟场"}],"zh":"粒子图"},{"status":"planned","key":"roadmap.galaxy.animation.branch.3","en":"STATE MACHINE","leaves":[{"status":"planned","key":"roadmap.galaxy.animation.leaf.3.0","en":"Transition graph","zh":"转换图"},{"status":"planned","key":"roadmap.galaxy.animation.leaf.3.1","en":"Blend spaces","zh":"混合空间"},{"status":"future","key":"roadmap.galaxy.animation.leaf.3.2","en":"Animation events","zh":"动画事件"},{"status":"future","key":"roadmap.galaxy.animation.leaf.3.3","en":"Authoring preview","zh":"创作预览"}],"zh":"状态机"}],"titleZh":"动画系统"},{"key":"physics","titleKey":"roadmap.galaxy.physics.title","branches":[{"status":"live","key":"roadmap.galaxy.physics.branch.0","en":"RIGID BODIES","leaves":[{"status":"live","key":"roadmap.galaxy.physics.leaf.0.0","en":"Jolt integration","zh":"Jolt 集成"},{"status":"live","key":"roadmap.galaxy.physics.leaf.0.1","en":"Colliders","zh":"碰撞体"},{"status":"planned","key":"roadmap.galaxy.physics.leaf.0.2","en":"Constraints","zh":"约束"},{"status":"future","key":"roadmap.galaxy.physics.leaf.0.3","en":"Vehicles","zh":"载具"}],"zh":"刚体"},{"status":"live","key":"roadmap.galaxy.physics.branch.1","en":"SCENE QUERIES","leaves":[{"status":"live","key":"roadmap.galaxy.physics.leaf.1.0","en":"Ray casts","zh":"射线检测"},{"status":"live","key":"roadmap.galaxy.physics.leaf.1.1","en":"Shape casts","zh":"形状检测"},{"status":"planned","key":"roadmap.galaxy.physics.leaf.1.2","en":"Overlap queries","zh":"重叠查询"},{"status":"planned","key":"roadmap.galaxy.physics.leaf.1.3","en":"Query filters","zh":"查询过滤"}],"zh":"场景查询"},{"status":"live","key":"roadmap.galaxy.physics.branch.2","en":"COLLISION","leaves":[{"status":"live","key":"roadmap.galaxy.physics.leaf.2.0","en":"Layer masks","zh":"层遮罩"},{"status":"live","key":"roadmap.galaxy.physics.leaf.2.1","en":"Contact callbacks","zh":"接触回调"},{"status":"planned","key":"roadmap.galaxy.physics.leaf.2.2","en":"Trigger volumes","zh":"触发体积"},{"status":"future","key":"roadmap.galaxy.physics.leaf.2.3","en":"Destruction","zh":"破坏系统"}],"zh":"碰撞"},{"status":"live","key":"roadmap.galaxy.physics.branch.3","en":"FIXED STEP","leaves":[{"status":"live","key":"roadmap.galaxy.physics.leaf.3.0","en":"Stable tick","zh":"稳定 Tick"},{"status":"planned","key":"roadmap.galaxy.physics.leaf.3.1","en":"Interpolation","zh":"插值"},{"status":"future","key":"roadmap.galaxy.physics.leaf.3.2","en":"Rollback hooks","zh":"回滚钩子"},{"status":"future","key":"roadmap.galaxy.physics.leaf.3.3","en":"Deterministic replay","zh":"确定性回放"}],"zh":"定步循环"}],"titleZh":"物理系统"},{"key":"audio","titleKey":"roadmap.galaxy.audio.title","branches":[{"status":"planned","key":"roadmap.galaxy.audio.branch.0","en":"SPATIAL AUDIO","leaves":[{"status":"planned","key":"roadmap.galaxy.audio.leaf.0.0","en":"Listener graph","zh":"监听器图"},{"status":"planned","key":"roadmap.galaxy.audio.leaf.0.1","en":"3D emitters","zh":"3D 发射器"},{"status":"future","key":"roadmap.galaxy.audio.leaf.0.2","en":"Occlusion","zh":"遮挡"},{"status":"future","key":"roadmap.galaxy.audio.leaf.0.3","en":"Room reverb","zh":"房间混响"}],"zh":"空间音频"},{"status":"planned","key":"roadmap.galaxy.audio.branch.1","en":"MIXER","leaves":[{"status":"planned","key":"roadmap.galaxy.audio.leaf.1.0","en":"Bus routing","zh":"总线路由"},{"status":"future","key":"roadmap.galaxy.audio.leaf.1.1","en":"Snapshots","zh":"快照"},{"status":"future","key":"roadmap.galaxy.audio.leaf.1.2","en":"Effects chain","zh":"效果链"},{"status":"future","key":"roadmap.galaxy.audio.leaf.1.3","en":"Metering","zh":"电平监视"}],"zh":"混音器"},{"status":"planned","key":"roadmap.galaxy.audio.branch.2","en":"STREAMING","leaves":[{"status":"planned","key":"roadmap.galaxy.audio.leaf.2.0","en":"Asset decode","zh":"资产解码"},{"status":"future","key":"roadmap.galaxy.audio.leaf.2.1","en":"Music playlists","zh":"音乐播放列表"},{"status":"future","key":"roadmap.galaxy.audio.leaf.2.2","en":"Voice capture","zh":"语音捕获"},{"status":"future","key":"roadmap.galaxy.audio.leaf.2.3","en":"Async loading","zh":"异步加载"}],"zh":"流式音频"},{"status":"future","key":"roadmap.galaxy.audio.branch.3","en":"CAPTURE","leaves":[{"status":"future","key":"roadmap.galaxy.audio.leaf.3.0","en":"Offline render","zh":"离线渲染"},{"status":"future","key":"roadmap.galaxy.audio.leaf.3.1","en":"Replay audio","zh":"回放音频"},{"status":"future","key":"roadmap.galaxy.audio.leaf.3.2","en":"Wave export","zh":"波形导出"},{"status":"future","key":"roadmap.galaxy.audio.leaf.3.3","en":"Diagnostics","zh":"诊断"}],"zh":"捕获"}],"titleZh":"音频"},{"key":"gameplay","titleKey":"roadmap.galaxy.gameplay.title","branches":[{"status":"live","key":"roadmap.galaxy.gameplay.branch.0","en":"COMPONENTS","leaves":[{"status":"live","key":"roadmap.galaxy.gameplay.leaf.0.0","en":"Python components","zh":"Python 组件"},{"status":"live","key":"roadmap.galaxy.gameplay.leaf.0.1","en":"Serialized fields","zh":"序列化字段"},{"status":"live","key":"roadmap.galaxy.gameplay.leaf.0.2","en":"Lifecycle hooks","zh":"生命周期钩子"},{"status":"planned","key":"roadmap.galaxy.gameplay.leaf.0.3","en":"Hot reload","zh":"热重载"}],"zh":"组件"},{"status":"live","key":"roadmap.galaxy.gameplay.branch.1","en":"INPUT","leaves":[{"status":"live","key":"roadmap.galaxy.gameplay.leaf.1.0","en":"Action maps","zh":"动作映射"},{"status":"planned","key":"roadmap.galaxy.gameplay.leaf.1.1","en":"Device routing","zh":"设备路由"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.1.2","en":"Rebinding","zh":"按键重绑定"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.1.3","en":"Input replay","zh":"输入回放"}],"zh":"输入"},{"status":"live","key":"roadmap.galaxy.gameplay.branch.2","en":"WORLD UI","leaves":[{"status":"live","key":"roadmap.galaxy.gameplay.leaf.2.0","en":"Canvas widgets","zh":"Canvas 控件"},{"status":"planned","key":"roadmap.galaxy.gameplay.leaf.2.1","en":"Layout system","zh":"布局系统"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.2.2","en":"UI animation","zh":"UI 动画"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.2.3","en":"Accessibility","zh":"无障碍支持"}],"zh":"世界 UI"},{"status":"planned","key":"roadmap.galaxy.gameplay.branch.3","en":"MULTIPLAYER","leaves":[{"status":"planned","key":"roadmap.galaxy.gameplay.leaf.3.0","en":"Replicated state","zh":"复制状态"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.3.1","en":"Authority model","zh":"权威模型"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.3.2","en":"Prediction","zh":"预测"},{"status":"future","key":"roadmap.galaxy.gameplay.leaf.3.3","en":"Session flow","zh":"会话流程"}],"zh":"多人游戏"}],"titleZh":"GamePlay"},{"key":"toolchain","titleKey":"roadmap.galaxy.toolchain.title","branches":[{"status":"live","key":"roadmap.galaxy.toolchain.branch.0","en":"EDITOR","leaves":[{"status":"live","key":"roadmap.galaxy.toolchain.leaf.0.0","en":"Scene editing","zh":"场景编辑"},{"status":"live","key":"roadmap.galaxy.toolchain.leaf.0.1","en":"Inspector","zh":"Inspector"},{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.0.2","en":"Command palette","zh":"命令面板"},{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.0.3","en":"Extension API","zh":"扩展 API"}],"zh":"编辑器"},{"status":"live","key":"roadmap.galaxy.toolchain.branch.1","en":"PACKAGES","leaves":[{"status":"live","key":"roadmap.galaxy.toolchain.leaf.1.0","en":"InxPackage","zh":"InxPackage"},{"status":"live","key":"roadmap.galaxy.toolchain.leaf.1.1","en":"Asset index","zh":"资产索引"},{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.1.2","en":"Plugin lifecycle","zh":"插件生命周期"},{"status":"future","key":"roadmap.galaxy.toolchain.leaf.1.3","en":"Dependency graph","zh":"依赖图"}],"zh":"包系统"},{"status":"live","key":"roadmap.galaxy.toolchain.branch.2","en":"HUB","leaves":[{"status":"live","key":"roadmap.galaxy.toolchain.leaf.2.0","en":"Environment setup","zh":"环境配置"},{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.2.1","en":"Artifact channels","zh":"构建产物通道"},{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.2.2","en":"Version pinning","zh":"版本固定"},{"status":"future","key":"roadmap.galaxy.toolchain.leaf.2.3","en":"Team workspaces","zh":"团队工作区"}],"zh":"Hub"},{"status":"planned","key":"roadmap.galaxy.toolchain.branch.3","en":"MCP TOOLS","leaves":[{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.3.0","en":"World inspection","zh":"世界检查"},{"status":"planned","key":"roadmap.galaxy.toolchain.leaf.3.1","en":"Safe mutations","zh":"安全变更"},{"status":"future","key":"roadmap.galaxy.toolchain.leaf.3.2","en":"Undo boundaries","zh":"撤销边界"},{"status":"future","key":"roadmap.galaxy.toolchain.leaf.3.3","en":"Agent sessions","zh":"Agent 会话"}],"zh":"MCP 工具"}],"titleZh":"工具链"},{"key":"network","titleKey":"roadmap.galaxy.network.title","branches":[{"status":"future","key":"roadmap.galaxy.network.branch.0","en":"TRANSPORT","leaves":[{"status":"future","key":"roadmap.galaxy.network.leaf.0.0","en":"Socket layer","zh":"Socket 层"},{"status":"future","key":"roadmap.galaxy.network.leaf.0.1","en":"Reliable channels","zh":"可靠通道"},{"status":"future","key":"roadmap.galaxy.network.leaf.0.2","en":"Datagrams","zh":"数据报"},{"status":"future","key":"roadmap.galaxy.network.leaf.0.3","en":"Encryption","zh":"加密"}],"zh":"传输"},{"status":"future","key":"roadmap.galaxy.network.branch.1","en":"REPLICATION","leaves":[{"status":"future","key":"roadmap.galaxy.network.leaf.1.0","en":"Snapshot delta","zh":"快照增量"},{"status":"future","key":"roadmap.galaxy.network.leaf.1.1","en":"Entity ownership","zh":"实体所有权"},{"status":"future","key":"roadmap.galaxy.network.leaf.1.2","en":"Interest filters","zh":"兴趣过滤"},{"status":"future","key":"roadmap.galaxy.network.leaf.1.3","en":"Bandwidth budget","zh":"带宽预算"}],"zh":"复制"},{"status":"future","key":"roadmap.galaxy.network.branch.2","en":"SERVER","leaves":[{"status":"future","key":"roadmap.galaxy.network.leaf.2.0","en":"Dedicated runtime","zh":"专用运行时"},{"status":"future","key":"roadmap.galaxy.network.leaf.2.1","en":"Headless player","zh":"无窗口 Player"},{"status":"future","key":"roadmap.galaxy.network.leaf.2.2","en":"Match sessions","zh":"匹配会话"},{"status":"future","key":"roadmap.galaxy.network.leaf.2.3","en":"Process supervision","zh":"进程监管"}],"zh":"服务器"},{"status":"future","key":"roadmap.galaxy.network.branch.3","en":"DIAGNOSTICS","leaves":[{"status":"future","key":"roadmap.galaxy.network.leaf.3.0","en":"Latency traces","zh":"延迟追踪"},{"status":"future","key":"roadmap.galaxy.network.leaf.3.1","en":"Network replay","zh":"网络回放"},{"status":"future","key":"roadmap.galaxy.network.leaf.3.2","en":"Packet inspection","zh":"数据包检查"},{"status":"future","key":"roadmap.galaxy.network.leaf.3.3","en":"Connection health","zh":"连接健康度"}],"zh":"诊断"}],"titleZh":"网络"},{"key":"data","titleKey":"roadmap.galaxy.data.title","branches":[{"status":"live","key":"roadmap.galaxy.data.branch.0","en":"BATCH ACCESS","leaves":[{"status":"live","key":"roadmap.galaxy.data.leaf.0.0","en":"World reads","zh":"世界读取"},{"status":"live","key":"roadmap.galaxy.data.leaf.0.1","en":"World writes","zh":"世界写入"},{"status":"planned","key":"roadmap.galaxy.data.leaf.0.2","en":"Column views","zh":"列视图"},{"status":"future","key":"roadmap.galaxy.data.leaf.0.3","en":"Dirty ranges","zh":"脏区间"}],"zh":"批量访问"},{"status":"live","key":"roadmap.galaxy.data.branch.1","en":"ARRAY BRIDGE","leaves":[{"status":"live","key":"roadmap.galaxy.data.leaf.1.0","en":"NumPy bridge","zh":"NumPy 桥"},{"status":"planned","key":"roadmap.galaxy.data.leaf.1.1","en":"Typed buffers","zh":"类型化缓冲"},{"status":"future","key":"roadmap.galaxy.data.leaf.1.2","en":"Zero-copy views","zh":"零拷贝视图"},{"status":"future","key":"roadmap.galaxy.data.leaf.1.3","en":"Memory layout","zh":"内存布局"}],"zh":"数组桥"},{"status":"planned","key":"roadmap.galaxy.data.branch.2","en":"JIT TARGETS","leaves":[{"status":"planned","key":"roadmap.galaxy.data.leaf.2.0","en":"CPU kernels","zh":"CPU Kernel"},{"status":"planned","key":"roadmap.galaxy.data.leaf.2.1","en":"GPU kernels","zh":"GPU Kernel"},{"status":"planned","key":"roadmap.galaxy.data.leaf.2.2","en":"Taichi bridge","zh":"Taichi 桥"},{"status":"future","key":"roadmap.galaxy.data.leaf.2.3","en":"Warmup cache","zh":"预热缓存"}],"zh":"JIT 目标"},{"status":"future","key":"roadmap.galaxy.data.branch.3","en":"TENSOR PLANE","leaves":[{"status":"future","key":"roadmap.galaxy.data.leaf.3.0","en":"Observation tensors","zh":"观测张量"},{"status":"future","key":"roadmap.galaxy.data.leaf.3.1","en":"Replay datasets","zh":"回放数据集"},{"status":"future","key":"roadmap.galaxy.data.leaf.3.2","en":"Batch stepping","zh":"批量步进"},{"status":"future","key":"roadmap.galaxy.data.leaf.3.3","en":"Inference buffers","zh":"推理缓冲"}],"zh":"张量数据面"}],"titleZh":"面向数据编程"}];
const ROADMAP_I18N_KEYS = ["roadmap.galaxy.foundation.title","roadmap.galaxy.foundation.branch.0","roadmap.galaxy.foundation.leaf.0.0","roadmap.galaxy.foundation.leaf.0.1","roadmap.galaxy.foundation.leaf.0.2","roadmap.galaxy.foundation.leaf.0.3","roadmap.galaxy.foundation.branch.1","roadmap.galaxy.foundation.leaf.1.0","roadmap.galaxy.foundation.leaf.1.1","roadmap.galaxy.foundation.leaf.1.2","roadmap.galaxy.foundation.leaf.1.3","roadmap.galaxy.foundation.branch.2","roadmap.galaxy.foundation.leaf.2.0","roadmap.galaxy.foundation.leaf.2.1","roadmap.galaxy.foundation.leaf.2.2","roadmap.galaxy.foundation.leaf.2.3","roadmap.galaxy.foundation.branch.3","roadmap.galaxy.foundation.leaf.3.0","roadmap.galaxy.foundation.leaf.3.1","roadmap.galaxy.foundation.leaf.3.2","roadmap.galaxy.foundation.leaf.3.3","roadmap.galaxy.rendering.title","roadmap.galaxy.rendering.branch.0","roadmap.galaxy.rendering.leaf.0.0","roadmap.galaxy.rendering.leaf.0.1","roadmap.galaxy.rendering.leaf.0.2","roadmap.galaxy.rendering.leaf.0.3","roadmap.galaxy.rendering.branch.1","roadmap.galaxy.rendering.leaf.1.0","roadmap.galaxy.rendering.leaf.1.1","roadmap.galaxy.rendering.leaf.1.2","roadmap.galaxy.rendering.leaf.1.3","roadmap.galaxy.rendering.branch.2","roadmap.galaxy.rendering.leaf.2.0","roadmap.galaxy.rendering.leaf.2.1","roadmap.galaxy.rendering.leaf.2.2","roadmap.galaxy.rendering.leaf.2.3","roadmap.galaxy.rendering.branch.3","roadmap.galaxy.rendering.leaf.3.0","roadmap.galaxy.rendering.leaf.3.1","roadmap.galaxy.rendering.leaf.3.2","roadmap.galaxy.rendering.leaf.3.3","roadmap.galaxy.animation.title","roadmap.galaxy.animation.branch.0","roadmap.galaxy.animation.leaf.0.0","roadmap.galaxy.animation.leaf.0.1","roadmap.galaxy.animation.leaf.0.2","roadmap.galaxy.animation.leaf.0.3","roadmap.galaxy.animation.branch.1","roadmap.galaxy.animation.leaf.1.0","roadmap.galaxy.animation.leaf.1.1","roadmap.galaxy.animation.leaf.1.2","roadmap.galaxy.animation.leaf.1.3","roadmap.galaxy.animation.branch.2","roadmap.galaxy.animation.leaf.2.0","roadmap.galaxy.animation.leaf.2.1","roadmap.galaxy.animation.leaf.2.2","roadmap.galaxy.animation.leaf.2.3","roadmap.galaxy.animation.branch.3","roadmap.galaxy.animation.leaf.3.0","roadmap.galaxy.animation.leaf.3.1","roadmap.galaxy.animation.leaf.3.2","roadmap.galaxy.animation.leaf.3.3","roadmap.galaxy.physics.title","roadmap.galaxy.physics.branch.0","roadmap.galaxy.physics.leaf.0.0","roadmap.galaxy.physics.leaf.0.1","roadmap.galaxy.physics.leaf.0.2","roadmap.galaxy.physics.leaf.0.3","roadmap.galaxy.physics.branch.1","roadmap.galaxy.physics.leaf.1.0","roadmap.galaxy.physics.leaf.1.1","roadmap.galaxy.physics.leaf.1.2","roadmap.galaxy.physics.leaf.1.3","roadmap.galaxy.physics.branch.2","roadmap.galaxy.physics.leaf.2.0","roadmap.galaxy.physics.leaf.2.1","roadmap.galaxy.physics.leaf.2.2","roadmap.galaxy.physics.leaf.2.3","roadmap.galaxy.physics.branch.3","roadmap.galaxy.physics.leaf.3.0","roadmap.galaxy.physics.leaf.3.1","roadmap.galaxy.physics.leaf.3.2","roadmap.galaxy.physics.leaf.3.3","roadmap.galaxy.audio.title","roadmap.galaxy.audio.branch.0","roadmap.galaxy.audio.leaf.0.0","roadmap.galaxy.audio.leaf.0.1","roadmap.galaxy.audio.leaf.0.2","roadmap.galaxy.audio.leaf.0.3","roadmap.galaxy.audio.branch.1","roadmap.galaxy.audio.leaf.1.0","roadmap.galaxy.audio.leaf.1.1","roadmap.galaxy.audio.leaf.1.2","roadmap.galaxy.audio.leaf.1.3","roadmap.galaxy.audio.branch.2","roadmap.galaxy.audio.leaf.2.0","roadmap.galaxy.audio.leaf.2.1","roadmap.galaxy.audio.leaf.2.2","roadmap.galaxy.audio.leaf.2.3","roadmap.galaxy.audio.branch.3","roadmap.galaxy.audio.leaf.3.0","roadmap.galaxy.audio.leaf.3.1","roadmap.galaxy.audio.leaf.3.2","roadmap.galaxy.audio.leaf.3.3","roadmap.galaxy.gameplay.title","roadmap.galaxy.gameplay.branch.0","roadmap.galaxy.gameplay.leaf.0.0","roadmap.galaxy.gameplay.leaf.0.1","roadmap.galaxy.gameplay.leaf.0.2","roadmap.galaxy.gameplay.leaf.0.3","roadmap.galaxy.gameplay.branch.1","roadmap.galaxy.gameplay.leaf.1.0","roadmap.galaxy.gameplay.leaf.1.1","roadmap.galaxy.gameplay.leaf.1.2","roadmap.galaxy.gameplay.leaf.1.3","roadmap.galaxy.gameplay.branch.2","roadmap.galaxy.gameplay.leaf.2.0","roadmap.galaxy.gameplay.leaf.2.1","roadmap.galaxy.gameplay.leaf.2.2","roadmap.galaxy.gameplay.leaf.2.3","roadmap.galaxy.gameplay.branch.3","roadmap.galaxy.gameplay.leaf.3.0","roadmap.galaxy.gameplay.leaf.3.1","roadmap.galaxy.gameplay.leaf.3.2","roadmap.galaxy.gameplay.leaf.3.3","roadmap.galaxy.toolchain.title","roadmap.galaxy.toolchain.branch.0","roadmap.galaxy.toolchain.leaf.0.0","roadmap.galaxy.toolchain.leaf.0.1","roadmap.galaxy.toolchain.leaf.0.2","roadmap.galaxy.toolchain.leaf.0.3","roadmap.galaxy.toolchain.branch.1","roadmap.galaxy.toolchain.leaf.1.0","roadmap.galaxy.toolchain.leaf.1.1","roadmap.galaxy.toolchain.leaf.1.2","roadmap.galaxy.toolchain.leaf.1.3","roadmap.galaxy.toolchain.branch.2","roadmap.galaxy.toolchain.leaf.2.0","roadmap.galaxy.toolchain.leaf.2.1","roadmap.galaxy.toolchain.leaf.2.2","roadmap.galaxy.toolchain.leaf.2.3","roadmap.galaxy.toolchain.branch.3","roadmap.galaxy.toolchain.leaf.3.0","roadmap.galaxy.toolchain.leaf.3.1","roadmap.galaxy.toolchain.leaf.3.2","roadmap.galaxy.toolchain.leaf.3.3","roadmap.galaxy.network.title","roadmap.galaxy.network.branch.0","roadmap.galaxy.network.leaf.0.0","roadmap.galaxy.network.leaf.0.1","roadmap.galaxy.network.leaf.0.2","roadmap.galaxy.network.leaf.0.3","roadmap.galaxy.network.branch.1","roadmap.galaxy.network.leaf.1.0","roadmap.galaxy.network.leaf.1.1","roadmap.galaxy.network.leaf.1.2","roadmap.galaxy.network.leaf.1.3","roadmap.galaxy.network.branch.2","roadmap.galaxy.network.leaf.2.0","roadmap.galaxy.network.leaf.2.1","roadmap.galaxy.network.leaf.2.2","roadmap.galaxy.network.leaf.2.3","roadmap.galaxy.network.branch.3","roadmap.galaxy.network.leaf.3.0","roadmap.galaxy.network.leaf.3.1","roadmap.galaxy.network.leaf.3.2","roadmap.galaxy.network.leaf.3.3","roadmap.galaxy.data.title","roadmap.galaxy.data.branch.0","roadmap.galaxy.data.leaf.0.0","roadmap.galaxy.data.leaf.0.1","roadmap.galaxy.data.leaf.0.2","roadmap.galaxy.data.leaf.0.3","roadmap.galaxy.data.branch.1","roadmap.galaxy.data.leaf.1.0","roadmap.galaxy.data.leaf.1.1","roadmap.galaxy.data.leaf.1.2","roadmap.galaxy.data.leaf.1.3","roadmap.galaxy.data.branch.2","roadmap.galaxy.data.leaf.2.0","roadmap.galaxy.data.leaf.2.1","roadmap.galaxy.data.leaf.2.2","roadmap.galaxy.data.leaf.2.3","roadmap.galaxy.data.branch.3","roadmap.galaxy.data.leaf.3.0","roadmap.galaxy.data.leaf.3.1","roadmap.galaxy.data.leaf.3.2","roadmap.galaxy.data.leaf.3.3","roadmap.hero.title","roadmap.changelog.title","roadmap.graph.hint","roadmap.graph.reset"];
/* Deterministic star maps: one galaxy per engine area, with GSAP motion and local pan/zoom. */
(function () {
    const app = document.querySelector("[data-roadmap-app]");
    if (!app) return;

    const tabs = Array.from(app.querySelectorAll("[data-tree-page]"));
    const pages = Array.from(app.querySelectorAll("[data-tree-panel]"));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const aliases = { architecture: "foundation", rendering: "pipeline", animation: "runtime", physics: "agents" };
    const cameras = new WeakMap();

    function defaultCamera(shell) {
        if (!window.matchMedia("(max-width: 820px)").matches) return { x: 0, y: 0 };
        const svg = shell.querySelector("svg");
        if (!svg) return { x: -420, y: 0 };
        const rect = svg.getBoundingClientRect();
        if (!rect.width) return { x: 0, y: 0 };
        const scale = rect.width / 1600;
        return { x: shell.clientWidth / (2 * scale) - 800, y: shell.clientHeight / (2 * scale) - 600 };
    }

    function cameraFor(shell) {
        if (!cameras.has(shell)) {
            const initial = defaultCamera(shell);
            cameras.set(shell, { ...initial, scale: 1, dragging: false, moved: false, initialized: false });
        }
        return cameras.get(shell);
    }

    function cameraTransform(camera) {
        return `translate(${camera.x} ${camera.y}) scale(${camera.scale})`;
    }

    function applyCamera(shell, smooth = false) {
        const camera = cameraFor(shell);
        const group = shell.querySelector("[data-graph-camera]");
        if (!group) return;
        const transform = cameraTransform(camera);
        if (smooth && !reduceMotion && globalThis.gsap) {
            globalThis.gsap.to(group, { attr: { transform }, duration: 0.24, ease: "power3.out", overwrite: "auto" });
        } else {
            group.setAttribute("transform", transform);
        }
    }

    function resetCamera(shell) {
        const camera = cameraFor(shell);
        const initial = defaultCamera(shell);
        camera.x = initial.x;
        camera.y = initial.y;
        camera.scale = 1;
        camera.initialized = true;
        applyCamera(shell, true);
    }

    function graphPoint(shell, event) {
        const rect = shell.querySelector("svg").getBoundingClientRect();
        return { x: ((event.clientX - rect.left) / rect.width) * 1600, y: ((event.clientY - rect.top) / rect.height) * 1200 };
    }

    function pageFromHash() {
        const raw = window.location.hash.replace(/^#tree-/, "");
        const value = aliases[raw] || raw;
        return tabs.some((tab) => tab.dataset.treePage === value) ? value : tabs[0]?.dataset.treePage;
    }


    const svgNs = "http://www.w3.org/2000/svg";

    function svgElement(tag, attributes = {}) {
        const element = document.createElementNS(svgNs, tag);
        Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
        return element;
    }

    function statusForBranches(branches) {
        if (branches.some((branch) => branch.status === "live")) return "live";
        if (branches.some((branch) => branch.status === "planned")) return "planned";
        return "future";
    }

    function renderGalaxy(panel) {
        if (!panel || panel.dataset.rendered === "true") return;
        const data = GALAXIES.find((galaxy) => galaxy.key === panel.dataset.galaxy);
        const svg = panel.querySelector(".node-graph");
        const cameraGroup = panel.querySelector("[data-graph-camera]");
        const starfield = panel.querySelector("[data-starfield]");
        if (!data || !svg || !cameraGroup || !starfield) return;
        const chinese = document.documentElement.lang.startsWith("zh");
        const label = (item) => chinese && item.zh ? item.zh : item.en;
        let state = 7001 + GALAXIES.indexOf(data) * 971;
        const random = () => {
            state = (1103515245 * state + 12345) & 0x7fffffff;
            return state / 0x7fffffff;
        };
        for (let index = 0; index < 72; index += 1) {
            starfield.append(svgElement("circle", {
                class: `galaxy-star${index % 13 === 0 ? " star-bright" : index % 7 === 0 ? " star-warm" : ""}`,
                cx: (38 + random() * 1524).toFixed(1),
                cy: (34 + random() * 1132).toFixed(1),
                r: [0.6, 0.8, 1.1, 1.6][Math.floor(random() * 4)],
            }));
        }
        cameraGroup.replaceChildren();
        const center = { x: 800, y: 600 };
        const angles = [-2.18, -0.72, 0.26, 1.54];
        const positions = data.branches.map((branch, index) => {
            const angle = angles[index] + (GALAXIES.indexOf(data) - 4) * 0.035 + ((GALAXIES.indexOf(data) + index) % 2 ? 0.08 : -0.04);
            const radius = 270 + ((GALAXIES.indexOf(data) * 29 + index * 37) % 70);
            return { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius * 0.79, angle };
        });
        const addLine = (from, to, status) => cameraGroup.append(svgElement("line", {
            class: `graph-edge state-${status}`, x1: from.x.toFixed(1), y1: from.y.toFixed(1), x2: to.x.toFixed(1), y2: to.y.toFixed(1),
        }));
        const bindNode = (node) => {
            node.setAttribute("aria-pressed", "false");
            node.addEventListener("click", () => selectNode(node));
            node.addEventListener("keydown", (event) => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                selectNode(node);
            });
        };
        const addNode = ({ type, id, status, key, en, zh, x, y, radius }) => {
            const node = svgElement("g", { class: `graph-node graph-${type} state-${status}`, tabindex: 0, role: "button", "data-node-id": id, "aria-label": en });
            node.append(svgElement("circle", { class: "graph-hit", cx: x.toFixed(1), cy: y.toFixed(1), r: type === "root" ? 34 : type === "branch" ? 30 : 25 }));
            node.append(svgElement("circle", { class: "graph-dot", cx: x.toFixed(1), cy: y.toFixed(1), r: radius }));
            const text = svgElement("text", { class: type === "leaf" ? "graph-leaf-label" : `graph-label${type === "root" ? " graph-root-label" : ""}`, x: x.toFixed(1), y: (y + (type === "root" ? 43 : 34)).toFixed(1), "text-anchor": "middle" });
            text.textContent = label({ en, zh });
            text.setAttribute("data-i18n", key);
            node.append(text);
            cameraGroup.append(node);
            bindNode(node);
        };
        data.branches.forEach((branch, branchIndex) => {
            const branchPosition = positions[branchIndex];
            addLine(center, branchPosition, branch.status);
            branch.leaves.forEach((leaf, leafIndex) => {
                const leafAngle = branchPosition.angle + (-0.82 + leafIndex * 0.53) + ((GALAXIES.indexOf(data) + branchIndex + leafIndex) % 3 - 1) * 0.08;
                const leafRadius = 152 + ((GALAXIES.indexOf(data) * 17 + branchIndex * 23 + leafIndex * 11) % 54);
                const leafPosition = { x: branchPosition.x + Math.cos(leafAngle) * leafRadius, y: branchPosition.y + Math.sin(leafAngle) * leafRadius * 0.78 };
                addLine(branchPosition, leafPosition, leaf.status);
                addNode({ type: "leaf", id: `${data.key}-${branchIndex}-${leafIndex}`, ...leaf, x: leafPosition.x, y: leafPosition.y, radius: 5 + (leafIndex % 3) });
            });
            addNode({ type: "branch", id: `${data.key}-${branchIndex}`, ...branch, x: branchPosition.x, y: branchPosition.y, radius: 9 });
        });
        addNode({ type: "root", id: `${data.key}-root`, status: statusForBranches(data.branches), key: data.titleKey, en: data.key.toUpperCase(), zh: data.titleZh, x: center.x, y: center.y, radius: 15 });
        panel.dataset.rendered = "true";
    }

    function animateGalaxy(panel) {
        if (reduceMotion || !globalThis.gsap || !panel) return;
        const stars = panel.querySelectorAll(".galaxy-star");
        const edges = panel.querySelectorAll(".graph-edge");
        const nodes = panel.querySelectorAll(".graph-node");
        globalThis.gsap.killTweensOf(stars);
        globalThis.gsap.killTweensOf(edges);
        globalThis.gsap.killTweensOf(nodes);
        globalThis.gsap.fromTo(stars, { autoAlpha: 0, scale: 0.2 }, { autoAlpha: 0.78, scale: 1, duration: 0.55, stagger: { each: 0.008, from: "random" }, ease: "power2.out", overwrite: "auto" });
        globalThis.gsap.to(stars, { autoAlpha: 0.28, duration: 1.6, stagger: { each: 0.035, from: "random" }, repeat: -1, yoyo: true, ease: "sine.inOut", overwrite: "auto" });
        globalThis.gsap.fromTo(edges, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.65, stagger: 0.018, ease: "power2.out", overwrite: "auto" });
        globalThis.gsap.fromTo(nodes, { autoAlpha: 0, y: 14 }, { autoAlpha: 1, y: 0, duration: 0.42, stagger: 0.018, ease: "power3.out", overwrite: "auto" });
    }

    function selectNode(node) {
        const panel = node.closest("[data-tree-panel]");
        if (!panel) return;
        panel.querySelectorAll(".graph-node").forEach((candidate) => {
            const selected = candidate === node;
            candidate.classList.toggle("is-selected", selected);
            candidate.setAttribute("aria-pressed", String(selected));
        });
        if (!reduceMotion && globalThis.gsap) {
            globalThis.gsap.fromTo(node.querySelector(".graph-dot"), { autoAlpha: 0.45 }, { autoAlpha: 1, duration: 0.3, ease: "back.out(2)", overwrite: "auto" });
        }
    }

    function selectPage(name, updateHistory = true) {
        const selected = tabs.find((tab) => tab.dataset.treePage === name) || tabs[0];
        if (!selected) return;
        const pageName = selected.dataset.treePage;
        tabs.forEach((tab) => {
            const active = tab === selected;
            tab.classList.toggle("is-active", active);
            tab.setAttribute("aria-selected", String(active));
        });
        pages.forEach((page) => {
            const active = page.dataset.treePanel === pageName;
            page.classList.toggle("is-active", active);
            page.hidden = !active;
        });
        if (updateHistory) history.replaceState(null, "", `#tree-${pageName}`);
        const activePage = pages.find((page) => page.dataset.treePanel === pageName);
        const activeShell = activePage?.querySelector("[data-graph-canvas]");
        if (activeShell) {
            renderGalaxy(activePage);
            const camera = cameraFor(activeShell);
            if (!camera.initialized) {
                const initial = defaultCamera(activeShell);
                camera.x = initial.x;
                camera.y = initial.y;
                camera.initialized = true;
            }
            applyCamera(activeShell);
            animateGalaxy(activePage);
        }
    }

    tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectPage(tab.dataset.treePage));
        tab.addEventListener("keydown", (event) => {
            const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1
                : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1
                    : event.key === "Home" ? -index
                        : event.key === "End" ? tabs.length - 1 - index : 0;
            if (!direction) return;
            event.preventDefault();
            const next = tabs[(index + direction + tabs.length) % tabs.length];
            next.focus();
            selectPage(next.dataset.treePage);
        });
    });

    app.querySelectorAll("[data-graph-canvas]").forEach((shell) => {
        const camera = cameraFor(shell);
        shell.addEventListener("pointerdown", (event) => {
            if (event.target.closest(".graph-node")) return;
            camera.dragging = true;
            camera.moved = false;
            camera.startX = event.clientX;
            camera.startY = event.clientY;
            camera.originX = camera.x;
            camera.originY = camera.y;
            shell.classList.add("is-panning");
            shell.setPointerCapture(event.pointerId);
        });
        shell.addEventListener("pointermove", (event) => {
            if (!camera.dragging) return;
            const rect = shell.querySelector("svg").getBoundingClientRect();
            const dx = (event.clientX - camera.startX) * 1600 / rect.width / camera.scale;
            const dy = (event.clientY - camera.startY) * 1200 / rect.height / camera.scale;
            camera.x = camera.originX + dx;
            camera.y = camera.originY + dy;
            camera.moved = Math.abs(dx) + Math.abs(dy) > 2;
            applyCamera(shell);
        });
        const endDrag = (event) => {
            if (!camera.dragging) return;
            camera.dragging = false;
            shell.classList.remove("is-panning");
            if (shell.hasPointerCapture(event.pointerId)) shell.releasePointerCapture(event.pointerId);
        };
        shell.addEventListener("pointerup", endDrag);
        shell.addEventListener("pointercancel", endDrag);
        shell.addEventListener("wheel", (event) => {
            event.preventDefault();
            const before = graphPoint(shell, event);
            const nextScale = Math.min(2.35, Math.max(0.62, camera.scale * (event.deltaY < 0 ? 1.1 : 0.9)));
            const ratio = nextScale / camera.scale;
            camera.x = before.x - (before.x - camera.x) * ratio;
            camera.y = before.y - (before.y - camera.y) * ratio;
            camera.scale = nextScale;
            applyCamera(shell, true);
        }, { passive: false });
    });

    app.querySelector("[data-graph-reset]")?.addEventListener("click", () => {
        const active = pages.find((page) => !page.hidden);
        const shell = active?.querySelector("[data-graph-canvas]");
        if (shell) resetCamera(shell);
    });

    window.addEventListener("hashchange", () => selectPage(pageFromHash(), false));
    selectPage(pageFromHash(), false);
}());
