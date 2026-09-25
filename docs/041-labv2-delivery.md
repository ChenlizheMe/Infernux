# Infernux 041 Labv2 交付说明

这份说明定义的是 **Infernux041Labv2 的公共能力演示项目**。它交付可复用的引擎 API、示例场景和验收入口，不包含 Runner Long 原游戏关卡迁移、商业素材、原游戏玩法流程或一次性迁移脚本。Taichi 是引擎内部的裁剪编译能力，项目不安装独立 Taichi 插件。

## 启动

在 Windows 开发机上：

```powershell
conda activate infernux
python dev/041_launch_labv2_current.py
```

编辑器启动后，MCP 服务默认在 `http://127.0.0.1:9713/mcp`。打开 `00_DemoHub`，用四个入口按钮进入 Jelly、CPU JIT、模型和 World UI 样本；`Reset Scene` 重新载入当前场景，参数滑条控制演示时间倍率。项目自带的 `mcp.json` 可直接提供该服务给支持 MCP 的客户端。

交付前先运行来源结构审计：

```powershell
conda activate infernux
python scripts/acceptance/labv2_delivery_audit.py `
  --project D:\Users\Chenlizhe\Desktop\Infernux041Labv2 `
  --output out\acceptance\labv2-source-audit.json
```

该命令只检查项目文件、GUID sidecar、脚本、核心资产和 BuildSettings。报告中的 `runtime_acceptance`、`platform_acceptance` 和 `commercial_asset_review` 会明确保持 `not_claimed`，来源文件存在不能替代实际 Editor/Player/设备验收。准备把所有记录在案的场景纳入 Player 时，再显式使用 `--require-build-scenes --strict`；当前作者阶段有些实验场景故意不在 BuildSettings，审计会给出警告而不会偷偷当作发行通过。

## 场景目录

| 入口 | 作用 | 主要操作 | 当前交付边界 |
| --- | --- | --- | --- |
| `00_DemoHub` | 统一入口 | 打开四个样本、返回、重置、调节时间倍率 | Editor 导航已验证；每个 Player/平台仍须使用对应证据报告 |
| `01_XPBD_Jelly` | GPU 驻留 XPBD 透明果冻 | `I` 施加向上冲量，`R` 重置；选中对象可查看内部 Gizmo | 无 CPU 回退；自碰撞、完整刚体阶梯和最终视觉发行矩阵仍是独立门禁 |
| `12_CurrentJit` | CPU JIT 作者 API | Play 后查看 Console 的 `INFERNUX_CURRENT_JIT_DEMO_READY` | 桌面可运行；Android/Web 按平台能力走普通 Python 或构建期诊断，不承诺设备端 JIT |
| `04_ModelSelection` | Blender/FBX/GLTF/OBJ 模型资源 | 选择模型节点，在 Inspector 查看 Mesh、材质和子资源 | 资源源文件只在 Editor 侧；Player 只消费 GUID Cook 产物 |
| `15_WorldUICameraPolicies` | World UI 相机和遮挡策略 | 在 Scene/Game 中观察深度、相机切换和文字 | 视觉证据按平台单独记录 |
| `02_LevelCatalog` | DataAsset 目录作者流程 | Play 后观察 `LEVEL_CATALOG_READY`，使用项目内作者快捷键编辑/重置 | 机制样本，不是完整游戏关卡数据 |
| `03_PublishedClassHandles` | 已发布脚本类和属性查询 | Play 后观察 `PROPERTY_LAB_READY`，按项目 README 操作属性覆盖 | 机制样本，不承诺旧工程迁移 |
| `01_PrefabProperties` / `06_PrefabVariants` | Prefab 与 Variant 作者流程 | 通过项目内 Editor 工具创建、Apply/Revert、Undo/Redo | 仍需按 A13 复合证据确认完整结构矩阵 |
| `05_ImportedAnimation` / `13_A16EditorSync` | 动画和 Blender 源同步 | 在 Inspector/Project 查看导入子资源和源更新 | 相机/灯光不作为模型导入语义 |
| `09_UserMaterialLesson` | 材质教学 | 按 Learn 课程修改材质并观察 Game | 课程样本，不等于完整发行场景 |
| `14_WorldTextClarity` / `Assets/Acceptance/A06UIShader/16_A06CustomUIShader` | 文字清晰度和自定义 UI Shader | Scene/Game 截帧对照 | 只使用项目内样本材质，平台像素证据单独归档 |

## 验收记录规则

每一次交付记录至少包含：项目提交、场景 GUID、目标平台、分辨率、质量设置、规模、固定步设置、首次载入和稳定运行帧时，以及 Console/Player/设备日志。连续捕获必须包含开始和交互后的状态；单张最佳截图不能替代运行证据。

平台状态分开记录：

- Windows Editor/Player：需有当前提交的 MCP/窗口操作、Scene/Game capture 和无错误退出记录。
- Linux Editor/Player：需记录真实 X11、Wayland 或 XWayland 会话，SSH 纯 tty 不算窗口验收。
- Android：需记录 ABI、Cook 诊断、安装包名 `com.infernux.infernux041labv2`、Vulkan surface 生命周期和设备日志；不支持 JIT 的构建必须把脚本作为普通版本 Cook。
- Web：需记录当前提交的浏览器 Player、ordinary-Python Cook、GPU/浏览器错误门禁和输入；Web 不携带 Numba/llvmlite，也不能用空计算模块代替 GPU kernel 能力。

没有对应平台的当前提交证据时，交付报告必须保留 `not_claimed` 或 `pending`，不能把另一个平台的截图或基础 fixture 的通过结果复制到这里。

## 已知限制

- Jelly 当前演示覆盖 GPU 计算、压缩/回弹和碰撞反馈；自碰撞、完整双向刚体耦合数量矩阵、P99 尾延迟和最终跨平台视觉仍由 A03.4/A15.2 独立验收。
- 雪地、轮廓、物理游乐场等计划入口可能只有作者场景或首版实现，审计脚本不会因为场景名称存在而宣称可发行。
- Blender/FBX/GLTF 等源模型由 Editor 导入；Player 包只包含 GUID 寻址的 Cook 资源，不应把 `Assets`、`Library` 或原始模型复制进发行目录。
- 资产许可、字体许可、音频许可必须在发布前由发行审查确认；本说明和来源审计不替代法律/素材清单。

最终发行仍须通过 A15.2 的 Player、安装、平台、CI、文档和制品门禁。来源审计是交付前的第一步，用来避免把一个可打开的作者工程误称为完整发行包。
