# 048 → 050 全量迁移计划

分支：`yanshi/automation-host`（基于 `050/recreate_rendering_physical`）
基线：050 `32e778773`；048 `5424d0e93`
原则：**冲突处保留 050 的实现，在 050 的结构上重做 048 的能力**；每块独立提交、独立验证。

## 迁移矩阵

| # | 048 内容 | 050 现状 | 策略 | 状态 |
|---|---|---|---|---|
| 1 | Automation Host（20 模块、108 操作） | 无 | 整体迁入，保留 050 的 shader 操作 | ✅ 已完成 |
| 2 | Automation SDK（`infernux_automation`） | 无 | 整体迁入 | ✅ 已完成 |
| 3 | `sim.step/clock/pose` + 物理步进接口 | 无 | 基于 050 的 `SceneManager.step` 原生暂停步进重做 | ✅ 已完成并实测 |
| 4 | 新增 Python 模块（reload、UI、插件事务、图标/字体） | 部分缺失 | 186 个新增文件整体迁入 | ✅ 已完成 |
| 5 | 插件安装事务 | 050 私有 `_InstallTransaction` | 改用 048 的共享实现，保留 050 调用点 | ✅ 已完成 |
| 6 | reload 归属模型（owner/admission） | 050 有脚本重载 + runtime world | 保留 050 重载路径，把 owner 绑定进 050 的模块发布/退休，旧回调抛 `ReferenceError` | 🚧 进行中 |
| 7 | 插件 registry ownership ledger（`publish_generated_members`） | 050 用 files 记录，无 ledger | 把 048 的 ledger 适配到 050 的 registry 记录结构，用于 Player 生成产物 | ⬜ 待做 |
| 8 | 主题贡献系统（`editor_themes`/`theme_data`/theme panel） | 050 只有内置主题表 | 贡献文档校验 + 目录 + 选择；应用时落到 050 的原生主题，token 覆盖留给 Python 绘制 | ⬜ 待做 |
| 9 | 编辑器视觉刷新（C++ 49 文件：字体/控件/面板样式） | 050 自己的编辑器演进 | 不整体替换；按特性（排版、字段绘制、组件卡片、主题绘制辅助）逐个移植并用截图验证 | ⬜ 待做（高风险） |
| 10 | locales en/zh 新增键 | 050 自己的 i18n 表 | 合并新增键，冲突键保留 050 文案 | ⬜ 待做 |
| 11 | 048 测试（reload/plugin/theme/host） | 部分不存在 | 逐个移植并按 050 行为调整，禁止删除 050 现有覆盖 | 🚧 进行中 |
| 12 | 048 文档/wiki 生成物 | — | 跳过生成物，只保留手写文档 | ⬜ 待做 |

## 关键约束

1. **不替换 050 的 `script_loader`**：050 的 runtime world（`_RuntimeWorldScope`、`runtime_module_name`、按世界退休）是 050 的核心能力，048 的 coordinator 不知道它。owner 模型只能作为附加的失效保护层。
2. **不整体替换 050 的编辑器 C++**：050 正在重写渲染/物理，编辑器面板与渲染器耦合；按特性移植并逐项截图验证。
3. **验收门槛**：每个迁移块必须通过
   - `python -m compileall` + 导入审计（`tools/yanshi/_import_audit.py`）
   - 相关 `tests/python` 子集
   - 编辑器/无头冒烟（headless probe 或 editor launch）
4. **提交粒度**：一块一个提交，提交信息写清"050 保留了什么、048 的意图如何实现"。

## 验证记录

- 自动化层：`tests/python/test_sim_contract.py` 5 passed；编辑器操作目录可构建（108 个操作）。
- Yanshi 任务卡：`tools/yanshi/rigid_diagnostics_probe.py` → x(10)=1.25 m，位置最大误差 4.8e-8 m，时间最大误差 2.3e-9 s，tick 逐点 +1。
- 插件：`test_plugin_updates.py` + `test_inxpackage_plugins.py` 51 passed。
