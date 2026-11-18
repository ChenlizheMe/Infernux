# Yanshi × Infernux：自动化层迁移说明（050 基线）

分支：`yanshi/automation-host`（基于 `050/recreate_rendering_physical`）
日期：2026-11-18

本文记录把 0.4.8 分支（`refactor/plugin-mcp-runtime`）中的自动化能力迁移到 050
时的取舍、接口契约和验证方式。目标不是把 0.4.8 全量合进来，而是让 050 具备
Yanshi 所需的世界绑定、能力发现、回执、证据与确定性步进，同时不破坏 050 正在
重写的渲染/物理/运行时世界。

## 1. 迁入的内容

| 组件 | 位置 | 说明 |
|---|---|---|
| Automation Host | `python/infernux/host/` | transport-neutral 操作注册表、schema、能力、job/query/report、HTTP bridge |
| 操作目录 | 同上（20 个模块） | 108 个操作、28 个能力：scene/asset/material/particle/camera/capture/input/runtime/player/plugin/theme/docs/console/ui/sim |
| Automation SDK | `python/infernux_automation/` | 薄客户端：`info/operations/schema/call/batch/submit/job/observe/report`，`transport.unknown_outcome` |
| Reload 归属 | `python/infernux/engine/reload/owners.py` | 供宿主操作记录 provider 生命周期；不接管 050 的脚本重载 |
| 确定性步进 | `python/infernux/sim.py`、`host/sim_operations.py` | Yanshi `sim.step@1` / `rigid.pose@1` 的引擎侧实现 |
| 宿主挂钩 | `host/bridge.py` | 运行时信号改接 050 的 `runtime_change_journal()` |

## 2. 没有迁入的内容（以及原因）

| 048 组件 | 原因 |
|---|---|
| `engine/reload/{coordinator,modules,components,source,adapters,engine_scopes}.py` | 它替换了 050 的脚本重载事务，但 050 的 `script_loader` 已引入 runtime world（`_RuntimeWorldScope`、`runtime_module_name`）；直接替换会回退 050 的多世界能力。050 现有重载路径继续负责脚本加载。 |
| `plugins/{manager,preload,registry,official,package}.py` 重写 | 依赖上面的 reload coordinator；050 的插件管理器工作正常。事务化安装与 ownership ledger 留待后续单独 PR。 |
| 编辑器主题刷新（C++ `function/editor/**`、`gui/**`、`InxRenderer*`、`ui/theme.py`、locales） | 与 050 的渲染重写正面冲突（75 文件 / 189 冲突块），且与 Yanshi 交付无关。`engine/editor_themes.py` 提供了兼容垫片，主题相关操作仍可用。 |
| MCP 传输插件源码 | 通过子模块指针指向 0.4.8 的 MCP 0.3.0（独立 Automation 服务），不并入主仓库。 |

## 3. Yanshi 契约映射

| Yanshi 能力 | Infernux 入口 | 模式 |
|---|---|---|
| `bind(create/attach)` | `infernux.project.info` + `scene.open`/`scene.loaded.get` | editor |
| `capabilities()` | `operations` + `schema`（能力名如 `sim.read`、`sim.write`、`scene.read`、`capture.write`） | 任意 |
| `rigid.pose@1` | headless：`infernux.sim.pose(obj)`；editor：`infernux.sim.pose` | 任意 |
| `sim.step@1` | headless：`infernux.sim.step(engine, expected_tick=k)`；editor：`infernux.sim.step` + `infernux.sim.clock` 轮询 | 手动/暂停 |
| `player.input` | `infernux.input.key` / `key.hold` / `pointer.*` | editor Play |
| `capture.game` | `infernux.capture.request{source:"game"}` → `capture.status{capture_id}` | editor Play |
| `scene.load_status` | `infernux.project.info.active_scene.last_load` / `document_state` | editor |
| 取消 | `job_cancel`（未开始作业）、`capture.cancel`（截图） | 任意 |
| 回执 | `call` 返回操作结果；`sim.step` 返回 `target_tick`；写后断连由客户端映射为 `transport.unknown_outcome` | 任意 |

### 3.1 步进语义

`infernux.sim.step` 推进的是**物理步**，不是游戏帧，走的是引擎原生的暂停步进通道
（`SceneManager.step`）：

- 请求的 `delta_time` 必须等于世界当前固定步长（否则 `sim.delta_mismatch`）；
- 世界必须处于 Play 且已暂停（否则 `sim.not_playing` / `sim.not_paused`）；
- `expected_tick` 与已发布物理步号不一致时拒绝（`sim.stale_tick`）；
- 一次请求必须恰好推进 1 步，否则 `sim.step_not_exact`；
- 调用方的 `engine.tick` 只负责完成这一次原生步进，不会混入第二次步进。

完整的手动驱动顺序（Play 进场的启动帧必须先排空，否则第一次步进会带上残留累加器）：

```python
engine = inx.Engine(inx.LogLevel.Warn, inx.RuntimeMode.Headless)
engine.init_headless(project)
# ... 建立夹具场景，进入 Play Mode ...
inx.sim.settle(engine)          # 排空 Play 启动帧
runtime_body = inx.sim.body("Slider")   # Play 用运行态克隆，必须重新解析
runtime_body.velocity = inx.Vector3(0.5, 0, 0)
inx.sim.pause()
base = inx.sim.clock()
for k in range(1, 11):
    inx.sim.step(engine, expected_tick=base["tick"] + k - 1)
    sample = inx.sim.pose(runtime_body)
```

实测（`tools/yanshi/rigid_diagnostics_probe.py`）：base tick=2、sim_s=0.02；
k=0..10 的 x 从 1.2000000 到 1.2500000，最大位置误差 4.8e-8 m，最大时间误差
2.3e-9 s，逐点 tick 严格 +1。评分时用 `sim_s - base["sim_s"]` 作为任务相对时间
（引擎在 Play 进场时已经推进了启动帧）。

前置条件：`ProjectSettings/PhysicsSettings.json` 的 `fixed_delta_time` 必须等于任务卡的 `dt`。

## 4. 验证

```powershell
# 单元/契约
python -m pytest tests/python/test_sim_contract.py -q

# 任务卡（headless，x(k)=1.2+0.005k，t=0.10s）
python tools/yanshi/rigid_diagnostics_probe.py
```

`tools/yanshi/rigid_diagnostics_probe.py` 构造无重力、无碰撞、无阻尼的单体，
以 0.5 m/s 沿 +X 运动，逐步采样并断言 0.1 mm / 1 μs 容差。

## 5. 与设计稿的已知缺口

1. **Edit/Play 对象重绑**：宿主查询返回运行态对象 id，但 050 的 runtime world
   会改变 id 语义；适配器需要在进入 Play 后重新解析对象（先用名称/路径）。
2. **场景加载冲突**：`project.info` 提供 `last_load` 与 `document_state`，但没有
   独立的"冲突文件清单"操作。
3. **能力门禁**：`OperationRegistry.execute` 支持 `capability_granted`，但 HTTP
   bridge 固定传 `"*"`；Yanshi 适配器需在写入前用 `operations()` 的
   `schema.capabilities` 做预检，并记录判定。
4. **receipt 统一包络**：写操作没有全局 `request_id/duration`；适配器需自行封装
   命令 ID 并持久化意图。
