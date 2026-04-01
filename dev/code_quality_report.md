# 代码质量更新报告

**分支**: `012/update_quality_of_code`
**基准**: `dev/baseline_audit.json` (合并前 master)
**总计**: 18 个文件修改, +757 / −1054 行 (净减 297 行)

---

## 提交记录

| # | 提交 | 说明 |
|---|------|------|
| 1 | `2950feb1` | **core+components**: 提取共享序列化辅助模块，消除魔数 |
| 2 | `95b1296a` | **engine/undo**: 提取辅助函数，去重游戏对象 ID 查找和选区捕获 |
| 3 | `5d719b76` | **engine/ui**: 提取组件右键菜单、颜色/纹理字段辅助 |
| 4 | `4950e3cd` | **hierarchy_panel**: 提取创建后初始化和画布父级查找辅助 |
| 5 | `a57ae148` | **inspector_components**: 统一 5 个 drop handler，提取资产引用字段辅助 |
| 6 | `98851aa3` | **toolbar_panel**: 数据驱动的相机参数循环 |
| 7 | `c8bc5131` | **renderstack**: 提取共享的后不透明管线拓扑 |
| 8 | `b0771528` | **components+ui**: 用字典查表替换派发阶梯，接入已有辅助函数 |

---

## 热点文件改善

| 文件 | 基准分 | 现分 | 变化 |
|------|--------|------|------|
| `engine/undo.py` | 3321 | 3118 | **−203** |
| `engine/ui/inspector_panel.py` | 2343 | 2181 | **−162** |
| `engine/ui/inspector_ui_components.py` | 1814 | 1600 | **−214** |
| `engine/ui/hierarchy_panel.py` | 1570 | 793 | **−777** |
| `engine/ui/inspector_components.py` | 1186 | 1030 | **−156** |
| `components/component.py` | 1055 | 802 | **−253** |
| `engine/ui/toolbar_panel.py` | 932 | ≤800 | **退出 Top 20** |

---

## 模块级改善

| 模块 | 基准分 | 现分 | 变化 |
|------|--------|------|------|
| **engine/ui** | 15633 | 13460 | **−2173** |
| **components** | 3186 | 2685 | **−501** |
| **engine** | 9587 | 9364 | **−223** |
| **ui** | 1306 | 1122 | **−184** |
| **renderstack** | 1412 | 1315 | **−97** |
| **合计 Python 发现** | **3318** | **3128** | **−190** |

---

## 主要改进内容

### 1. 消除重复模板代码

- **序列化辅助模块** (`components/_serialize_helpers.py`):
  将 `component.py` 和 `serializable_object.py` 中 8 种字典键引用类型的反序列化派发、
  空引用创建、资产引用序列化、向量序列化共计 ~200 行重复代码合并到共享模块。

- **组件右键菜单** (`inspector_panel.py`):
  2 处 ~55 行的右键菜单模板提取为 `_render_component_context_menu()` 和 `_remove_component()`。

- **颜色栏 & 纹理选择器** (`inspector_ui_components.py`):
  6 处颜色栏模板 → `_render_color_field()`；2 处 30 行纹理选择器 → `_render_texture_picker()`。

- **创建后初始化 + 画布父级** (`hierarchy_panel.py`):
  6 处 parent+select+record+notify 模板 → `_finalize_created_object()`；
  2 处 15 行画布祖先遍历 → `_find_canvas_parent_id()`。

- **拖放处理器统一** (`inspector_components.py`):
  5 个几乎相同的 `_apply_*_drop` 函数 → 通用 `_apply_reference_drop()`；
  4 个 MATERIAL/TEXTURE/SHADER/ASSET 字段渲染块 → `_render_asset_reference_field()` + `_ASSET_REF_CONFIG`。

- **渲染管线拓扑** (`renderstack`):
  forward 和 deferred 管线共享的 skybox→transparent→post-process 尾段 → `add_post_opaque_section()`。

- **UI 渲染派发** (`ui_render_dispatch.py`):
  4 个渲染函数中重复的 `getattr` 属性提取 → 接入已有的 `extract_common()` 和 `_extract_text_attrs()`。

### 2. 消除魔数

- `core/material.py`: Vulkan 混合常量 → `_VK_BLEND_SRC_ALPHA`, `_VK_BLEND_ONE_MINUS_SRC_ALPHA` 等命名常量；
  渲染队列 → `_RENDER_QUEUE_OPAQUE`, `_RENDER_QUEUE_TRANSPARENT`。
- `core/assets.py`: `_META_SUPPRESSION_TIMEOUT`, `_DEFAULT_DEBOUNCE_SEC`。

### 3. 降低面条感

- `engine/undo.py`: 提取 `_game_object_id_of()`（3 处 5 行块）、`_get_current_selection_ids()`（2 处相同方法）、`_safe_iter()`；`_invalidate_builtin_wrappers_for_object_tree` 嵌套深度 6→3。
- `toolbar_panel.py`: 99 行 `_popup_camera` → 35 行方法 + 8 行 `_CAMERA_PARAMS` 配置元组。
- `serialized_field._infer_field_type`: 53 分支 if/elif → 三个字典查找表 + 少量 isinstance 检查。

### 4. .pyi 桩文件更新

- `core/material.pyi`: 新增模块级常量声明。
- `core/assets.pyi`: 新增 `on_material_saved` 回调声明。

---

## 未来优化建议

| 优先级 | 文件/模块 | 建议 |
|--------|-----------|------|
| 高 | `inspector_ui_components._render_on_click_events` (393行) | 提取引用回调工厂和参数修改模板 |
| 中 | `project_panel.on_render_content` (296行) | 拆分为工具栏/面包屑、文件网格、键盘快捷键子方法 |
| 中 | `scene_view_panel` | 提取 Gizmo 交互逻辑到独立辅助类 |
| 低 | `components/builtin/*.py` | 为 `_cpp_component` 代理属性添加只读描述符工厂 |
| 低 | C++ 层 | SceneRenderer/RenderPassOutput 等高分文件的结构优化 |
