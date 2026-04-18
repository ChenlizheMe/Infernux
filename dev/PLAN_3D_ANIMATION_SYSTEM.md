# Infernux 3D 动画系统 — 现状、差距与实施计划

本文档在**当前仓库能力**（2D 动画管线、Assimp 网格导入、Lit 材质、无骨骼 GPU 数据）基础上，评估 **3D AnimClip、3D AnimFSM、3D 动画控制器、骨骼导入与配套服务** 的工作量，并给出分阶段路线图与风险说明。

---

## 1. 现状摘要（可作为 3D 扩展的「地基」）

| 领域 | 已有能力 | 对 3D 动画的意义 |
|------|-----------|------------------|
| **网格** | `InxMesh` + `MeshLoader`（Assimp）、`Vertex` 无骨骼权重字段 | 需扩展顶点布局与缓冲策略 |
| **渲染** | `MeshRenderer` 静态网格、多材质、PBR `lit.frag` | 需蒙皮绘制路径（骨骼矩阵 UBO/SSBO、可选双 pass） |
| **资源** | `.fbx/.obj/...` ModelImporter meta、`import_materials` | 骨骼可与同一 Assimp 读取路径复用；需新 meta / 侧车数据 |
| **2D 动画数据** | `AnimationClip`（`.animclip2d`：帧索引 + fps + loop） | 与 3D 曲线型 clip **不共用**文件格式，可复用命名与 AssetManager 模式 |
| **FSM 数据** | `AnimStateMachine`（`.animfsm`）、`mode: "2d" \| "3d"` 字段已存在 | **数据层**可扩展；编辑器与运行时需 3D 分支 |
| **2D 控制器** | `SpiritAnimator`（Python）驱动 `SpriteRenderer` | 3D 需新组件 + **C++ 侧每帧 pose 应用**（性能关键） |
| **编辑器** | `animclip2d_editor_panel`、`animfsm_editor_panel`、`node_graph_view` | 可复用图编辑 UX；3D clip 需时间轴 / 曲线 UI |

**结论**：数据上的 FSM 已预留 `3d`；**引擎内尚无骨骼、蒙皮、动画采样与 GPU 路径**，这是工作量的大头。

---

## 2. 目标范围（本计划覆盖）

1. **骨骼与绑定**：从 DCC（Blender 等）经 FBX/glTF 导入层次、bind pose、骨骼名稳定 ID、权重（多影响上限需定义，如 4）。
2. **AnimationClip3D**：每骨骼 TRS 或矩阵曲线、时间范围、循环、事件帧（可选二期）。
3. **AnimFSM 3D**：`mode=3d` 下状态引用 `animclip3d`、过渡混合时长、与 2D 同一表达式参数模型（已存在）。
4. **Animator3D（或 SpiritAnimator 的 3D 变体）**：每帧求当前 pose、写骨骼矩阵、驱动带 `SkinnedMeshRenderer`（或扩展 `MeshRenderer`）的对象。
5. **导入与工程服务**：`.meta` 或侧车 JSON 记录 skeletonId、clip 列表、重定向策略占位；与 AssetDatabase / 依赖图集成。

**明确不包含（可作为后续 Phase 2+）**：人形重定向（Humanoid）、IK、Timeline、Animation Rigging、物理 ragdoll 自动配对。

---

## 3. 差距分析（按技术栈）

### 3.1 C++ / 渲染（高工作量）

- **顶点格式**：当前 `Vertex` 无 `boneIndex`/`boneWeight`；需新版本或并行 `SkinnedVertex`，并贯穿 `MeshLoader`、GPU 上传、`SceneRenderer` 绘制分支。
- **蒙皮着色器**：`lit.frag` / `standard` vert 需骨骼矩阵输入（uniform buffer 或 storage buffer）；骨骼数量上限与 Vulkan 对齐规则。
- **骨骼运行时结构**：`Skeleton`、`Pose`、`LocalTransforms`、`GlobalMatrices`、缓存脏标记。
- **动画采样**：从 clip 资源在 CPU 插值（LERP/SLERP）→ 写 `boneMatrices`；多 clip 混合时双 pose 累加（FSM 过渡）。
- **与现有静态网格共存**：同场景内静态 + 蒙皮；材质系统已较完整，主要增量在 **vertex layout + descriptor set layout**。

**粗估**：**6–12 人周**（含调试、多平台、与现有渲染路径合并），视 Vulkan 抽象层改动量浮动。

### 3.2 Assimp 导入与资产（中高工作量）

- 在现有 `MeshLoader` / `ModelImporter` 旁增加 **骨骼与动画曲线** 扫描：  
  - `aiScene::mRootNode` 层次与 `aiBone` 权重写入扩展网格；  
  - `aiAnimation` 通道映射到引擎 bone 索引（名称哈希 + 稳定排序规则）。  
- **Clip 烘焙策略**：  
  - **A**：运行时从 FBX 按需采样（简单、包体小）；  
  - **B**：导入时烘焙为 `animclip3d` 二进制/JSON（编辑器友好、运行时快）。  
  建议 **B 为主**，与 2D「资源文件为真理」一致。

**粗估**：**3–6 人周**（含 Blender/FBX/glTF 差异处理、回归测试）。

### 3.3 Python 数据与编辑器（中工作量）

- **`AnimationClip3D` 数据类** + `.animclip3d`（或二进制 + 小 JSON 头）+ `AssetManager` 注册。
- **`AnimStateMachine` 3D**：`clip_guid` 已通用；校验 `mode=3d` 时 clip 扩展名与结构。
- **`animclip3d_editor_panel`**：曲线或关键帧表、时间轴、骨骼筛选；可先做 **「仅显示关键帧表 + 预览走 C++」** 的 MVP。
- **`animfsm_editor_panel`**：按 `mode` 切换节点颜色/校验；3D 状态预览需 C++ 视口或简化骨架线框。
- **`Animator3D` 组件**：序列化 `AnimStateMachineRef`、参数 API 与 `SpiritAnimator` 对齐；`update` 调 native 推进。

**粗估**：**4–8 人周**（MVP 偏 4，完整曲线编辑偏 8）。

### 3.4 绑定与 Python↔C++ 桥接（中工作量）

- `pybind11`：`SkinnedMeshRenderer` 或扩展 `MeshRenderer` 的 `sharedMesh` + `skeleton` 引用。
- 每帧：`Animator3D` 调 `native.step_fsm(dt)` 或 Python 驱动参数 + C++ 仅采样骨骼（折中方案性能较差，不推荐长期）。

**粗估**：**2–4 人周**。

### 3.5 测试与工具链（贯穿）

- 单元：骨骼索引映射、clip 时长、循环边界、混合权重。
- 集成：Blender 导出单一骨骼网格 + walk 动画；Play 模式与编辑器暂停一致性。
- 性能：N 角色同屏骨骼矩阵更新预算。

**粗估**：**2–4 人周**（与主开发并行）。

---

## 4. 总工作量估计（汇总）

| 板块 | 乐观 | 典型 | 保守 |
|------|------|------|------|
| C++ 渲染与蒙皮 | 6 pw | 9 pw | 12 pw |
| 导入与 clip 烘焙 | 3 pw | 4.5 pw | 6 pw |
| Python 数据 + Animator3D | 4 pw | 6 pw | 8 pw |
| 编辑器（clip3d + fsm3d） | 3 pw | 5 pw | 8 pw |
| 绑定 / 联调 / 测试 | 2 pw | 3 pw | 4 pw |
| **合计** | **~18 pw** | **~27.5 pw** | **~38 pw** |

换算：**约 4.5–9 个日历月** 的单全职引擎方向（按每月 4 有效周、且含需求波动）；若 **2 人并行（渲染 + 工具链）**，典型可压到 **~3–5 个月** 到可演示的 MVP（单角色 walk/idle + FSM 切换 + 基础蒙皮）。

---

## 5. 推荐分阶段路线图

### Phase 0 — 设计冻结（3–5 天）

- 定：`Vertex` 扩展 vs `SkinnedVertex` 新类型；骨骼上限；clip 文件格式（JSON 首版 vs MessagePack）；`MeshRenderer` vs 新 `SkinnedMeshRenderer`。
- 输出：ADR 短文（可放在 `dev/adr/`）+ 数据 schema 草图。

### Phase 1 — 骨骼与静态绑定网格（3–5 周）

- Assimp：读 bone、权重，写扩展网格与 `Skeleton` 资源（或 mesh 内嵌 bone palette）。
- C++：GPU 缓冲、descriptor、**无动画** 仅 bind pose 蒙皮验证（T-pose 正确即里程碑）。
- 绑定：`SkinnedMeshRenderer` + `Mesh` 引用同 GUID 或子资源。

### Phase 2 — 单 Clip 播放（2–4 周）

- 导入烘焙 `animclip3d`（每 bone 每帧或稀疏关键帧 + Hermite）。
- CPU 采样 → 骨骼矩阵 → 已有蒙皮管线。
- Python：`Animator3D` MVP，仅默认状态循环一个 clip。

### Phase 3 — AnimFSM 3D + 过渡混合（2–4 周）

- 复用 `AnimStateMachine` `mode=3d`；过渡区间双 pose blend。
- `SpiritAnimator` 逻辑在 C++ 或 Python 中抽「策略对象」，避免 duplicated FSM 代码。

### Phase 4 — 编辑器与 UX（3–6 周，可与 Phase 2/3 部分并行）

- `animclip3d` 面板：时间轴、关键帧编辑、从选中 FBX 提取 clip。
- `animfsm`：3D 校验、预览场景摄像机跟随包围盒。
- 资源菜单：Create AnimClip3D / Assign from Model。

### Phase 5 — 打磨与扩展（持续）

- 根运动（可选）、Animation Events、压缩（量化旋转）、LOD、Compute skinning。

---

## 6. 风险与依赖

| 风险 | 缓解 |
|------|------|
| Assimp 与 Blender 导出骨骼名/缩放轴不一致 | 导入时统一坐标系、记录 `scale_factor`、提供「重算 bind pose」工具 |
| Vulkan pipeline 变体爆炸 | 蒙皮用独立 `pipelineLayout`；静态网格路径不变 |
| Python FSM 每帧过重 | 核心采样与矩阵下放到 C++，`Animator3D` 只做参数与事件 |
| `.obj` 无骨骼 | 文档标明仅骨骼 FBX/glTF；OBJ 仍走静态 |

---

## 7. 建议的「最小可演示」里程碑（MVP 定义）

以下全部满足即可对外演示「3D 动画闭环」：

1. Blender 导出 **单材质** 蒙皮网格 + **一个** walk clip（FBX 或 glTF）。
2. 导入后场景内播放 walk，`AnimFSM` 在 idle/walk 两态切换（无曲线编辑，手改 JSON 亦可）。
3. Play 模式与编辑器下行为一致；帧率可接受（>30fps，1 角色）。

---

## 8. 文档与代码位置索引（实施时更新）

| 主题 | 参考路径 |
|------|-----------|
| 2D Clip | `python/Infernux/core/animation_clip.py` |
| FSM 数据 | `python/Infernux/core/anim_state_machine.py` |
| 2D 运行时 | `python/Infernux/components/animator2d.py` |
| 网格加载 | `cpp/infernux/function/resources/InxMesh/MeshLoader.cpp` |
| 顶点定义 | `cpp/infernux/function/renderer/InxRenderStruct.h` |
| Lit 着色 | `python/Infernux/resources/shaders/lit.frag` |

---

## 9. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-18 | 初版：基于仓库当前结构评估工作量与阶段划分 |

---

*本计划为工程规划文档，随实现进度在 `dev/` 内迭代修订。*
