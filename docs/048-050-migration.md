# 048 → 050 迁移记录

分支：`integration/048-050`（基于 `050/recreate_rendering_physical`）
基线：050 `32e778773`；048 `5424d0e93`
原则：**冲突处保留 050 的实现，在 050 的结构上重做 048 的能力**；每块独立提交、独立验证。

## 迁移矩阵

| # | 048 内容 | 050 现状 | 策略 | 状态 |
|---|---|---|---|---|
| 1 | Automation Host（20 模块、112 操作） | 无 | 整体迁入，保留 050 的 shader 操作 | ✅ |
| 2 | Automation SDK（`infernux_automation`） | 无 | 整体迁入 | ✅ |
| 3 | `sim.step/clock/pose` 物理步进接口 | 无 | 基于 050 的 `SceneManager.step` 原生暂停步进重做 | ✅ |
| 4 | 新增 Python 模块（reload、UI、插件事务、图标/字体） | 部分缺失 | 整体迁入 | ✅ |
| 5 | 插件安装事务 / manager / preload / official | 050 私有实现 + 本地修改保护 | **改用 048 全套**；移除 050 的本地修改冲突保护 | ✅ |
| 6 | 插件 registry ownership ledger | 050 用 files 记录，无 ledger | 048 registry 整体采用 | ✅ |
| 7 | reload 归属模型（owner/admission） | 050 loader 有 runtime world | 迁入 048 `engine/reload/*`、`script_dependency_graph`、`lifecycle`、`execution`、`runtime_services` 的 owner 语义，与 050 的 world 作用域合并 | ✅ |
| 8 | 组件脚本热重载 | 050 的 `prepare/commit_script_reload_batch` | 保留 050 路径；它按 `(authoring, *runtime_world_ids)` 逐世界 prepare/commit | ✅ 有回归测试 |
| 9 | 主题贡献系统（`editor_themes`/`theme_data`） | 050 只有内置主题表 | 050 registry 上新增 Definition/Register/RemoveOwner/ReplaceOwner + 绑定 | ✅ |
| 10 | 编辑器视觉刷新（C++ 49 文件） | 050 自己的编辑器演进 | 048 独占的绘制辅助/音频/采集已迁；面板与渲染器耦合部分待 GUI 对照 | 🚧 |
| 11 | locales en/zh 新增键 | 050 自己的 i18n 表 | 合并新增键，冲突键保留 050 文案 | ✅ |
| 12 | 048 测试 | 部分不存在 | 048 插件/主题测试已移植；仅依赖 048 loader 内部结构的 reload 测试不适用，已移除 | 🚧 |

## reload 的实际结构

- **包/预加载源重载**：048 的 `ReloadCoordinator` 负责 source 快照、依赖图、owner admission、retire/rebuild。`PluginPreloadManager` 通过它管理包生命周期。
- **组件脚本重载**：仍是 050 的 `ResourcesManager → PlayModeManager.prepare_script_reload_batch`。
  - 该方法对 `(authoring_runtime_world_id, *runtime_world_ids)` 逐个 `_RuntimeWorldScope` 预备事务；
  - 提交由 `WorldScriptPublication.commit()` 在同一组 world 作用域内完成，失败整体回滚；
  - 回归测试 `tests/python/test_play_mode_multi_world_reload.py` 固定了"每个 world 都 prepare、都 commit"的契约。
- **owner 语义合并**：`execution.py` 保留 050 的 world id / step permission / `value_world`，同时加入 048 的 `own_cleanup`、reload owner 进入/失效检查；`runtime_services.py` 保留 050 的按 world update 表，同时加入 048 的 registration/service owner、依赖准入与重载退休。

## 关键约束

1. 050 的 runtime world 是核心：任何重载/服务注册都必须在 world 作用域内工作。
2. 050 正在重写渲染/物理，编辑器面板与渲染器耦合；视觉刷新按特性移植并逐项截图验证。
3. 验收：`compileall` + 相关 `tests/python` 子集 + 构建 +（涉及编辑器时）启动冒烟。
4. 一块一个提交，提交信息写清"050 保留了什么、048 的意图如何实现"。

## 验证记录

- 插件：`test_plugin_updates` + `test_inxpackage_plugins` + `test_plugin_preload_packages` **76 passed**（048 插件栈）。
- reload：`test_play_mode_multi_world_reload` **1 passed**；`test_runtime_updates` / `test_play_mode_component_body_reload` / `test_sim_step` / `test_native_console` **84 passed**。
- 主题：`test_editor_theme_packages` 通过（按 050 的扁平 token 表调整断言）。
- 编辑器：048 插件栈 + reload bridge 启动成功（DoorScene2 加载、控制台干净、61 FPS）；`ThemePanel` 已注册进 050 的面板 registry。`n- 基线说明：`test_script_inheritance_reload`、`test_ui_font_reload`、`test_runtime_reload_stress_matrix` 在**干净的 050 worktree** 上同样失败（16/2/9），是 050 既有状态，不是本次迁移引入。

## 编辑器视觉刷新评估

- 048 的 `EditorThemeTable.inl` 有 **401** 个主题 token，050 有 **240** 个，且 050 的集合是 048 的子集——048 的刷新是同谱系增量。
- 050 在 `function/editor`、`renderer/gui` 上另有 4.3k 行改动，与 048 的面板刷新逐文件冲突；整体替换会回退 050 的编辑器工作。
- 已迁：`EditorThemeDrawing.h`、`EditorSurfacePattern.h`、主题贡献系统。待迁：048 的面板/排版刷新（需要 GUI 并排截图后逐项移植）。

## 下一步

1. 048 面板层（theme 已注册；catalog/settings/asset toolbar）挂载到 050 的 bootstrap，并做 GUI 交互冒烟。
2. 048 视觉刷新逐文件移植：先 `EditorThemeTable` 增量 token，再面板绘制。
3. 剩余 048 专属 UI 模块接线（`catalog_browser`、`settings_workspace`、`documentation_panel` 等）。
