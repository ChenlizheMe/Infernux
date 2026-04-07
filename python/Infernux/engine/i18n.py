"""
Internationalization (i18n) for the Infernux editor.

Provides a simple key-based translation system with two supported locales:
``"en"`` (English) and ``"zh"`` (Simplified Chinese).

Usage::

    from Infernux.engine.i18n import t

    label = t("menu.project")        # "Project" or "项目"
    label = t("menu.preferences")    # "Preferences" or "偏好设置"

The active locale is persisted to ``Documents/Infernux/preferences.json``
so it survives across sessions.
"""

from __future__ import annotations

import json
import os
import pathlib
from Infernux.debug import Debug

# ---------------------------------------------------------------------------
# Locale state
# ---------------------------------------------------------------------------

_current_locale: str = "zh"

# ---------------------------------------------------------------------------
# Translation table:  key  ->  { locale: text }
# ---------------------------------------------------------------------------

_STRINGS: dict[str, dict[str, str]] = {
    # ── Menu bar ──────────────────────────────────────────────────────
    "menu.project":                     {"en": "Project",                   "zh": "项目"},
    "menu.window":                      {"en": "Window",                    "zh": "窗口"},
    "menu.preferences":                 {"en": "Preferences",               "zh": "偏好设置"},
    "menu.build_settings":              {"en": "Build Settings",            "zh": "构建设置"},
    "menu.physics_layer_matrix":        {"en": "Physics Layer Matrix",      "zh": "物理层交互矩阵"},
    "menu.reset_layout":                {"en": "Reset Layout",              "zh": "重置布局"},
    "menu.no_windows":                  {"en": "(No windows registered)",   "zh": "(没有已注册窗口)"},
    "menu.no_wm":                       {"en": "(Window manager not set)",  "zh": "(窗口管理器未设置)"},

    # ── Panel titles ──────────────────────────────────────────────────
    "panel.hierarchy":                  {"en": "Hierarchy",                 "zh": "层级"},
    "panel.inspector":                  {"en": "Inspector",                 "zh": "检视器"},
    "panel.console":                    {"en": "Console",                   "zh": "控制台"},
    "panel.scene":                      {"en": "Scene",                     "zh": "场景"},
    "panel.game":                       {"en": "Game",                      "zh": "游戏"},
    "panel.project":                    {"en": "Project",                   "zh": "项目"},
    "panel.toolbar":                    {"en": "Toolbar",                   "zh": "工具栏"},
    "panel.tags_layers":                {"en": "Tags & Layers",             "zh": "标签与图层"},
    "panel.ui_editor":                  {"en": "UI Editor",                 "zh": "UI编辑器"},

    # ── Preferences window ────────────────────────────────────────────
    "prefs.title":                      {"en": "Preferences",               "zh": "偏好设置"},
    "prefs.language":                   {"en": "Language",                   "zh": "语言"},
    "prefs.language.en":                {"en": "English",                   "zh": "English"},
    "prefs.language.zh":                {"en": "简体中文",                   "zh": "简体中文"},

    # ── Build settings ────────────────────────────────────────────────
    "build.fullscreen_borderless":      {"en": "Fullscreen Borderless",     "zh": "全屏无边框"},
    "build.windowed":                   {"en": "Windowed",                  "zh": "窗口模式"},

    # ── Toolbar ───────────────────────────────────────────────────────
    "toolbar.play":                     {"en": "Play",                      "zh": "播放"},
    "toolbar.stop":                     {"en": "Stop",                      "zh": "停止"},
    "toolbar.pause":                    {"en": "Pause",                     "zh": "暂停"},
    "toolbar.resume":                   {"en": "Resume",                    "zh": "继续"},
    "toolbar.step":                     {"en": "Step",                      "zh": "步进"},
    "toolbar.status_paused":            {"en": "PAUSED",                    "zh": "已暂停"},
    "toolbar.status_playing":           {"en": "PLAYING",                   "zh": "运行中"},
    "toolbar.gizmos":                   {"en": "Gizmos  v",                 "zh": "Gizmos  v"},
    "toolbar.camera":                   {"en": "Camera  v",                 "zh": "Camera  v"},
    "toolbar.engine_not_available":     {"en": "Engine not available",      "zh": "引擎不可用"},
    "toolbar.gizmos_header":            {"en": "Gizmos",                    "zh": "Gizmos"},
    "toolbar.show_grid":                {"en": "Show Grid",                 "zh": "显示网格"},
    "toolbar.camera_not_available":     {"en": "Camera not available",      "zh": "摄像机不可用"},
    "toolbar.scene_camera":             {"en": "Scene Camera",              "zh": "场景摄像机"},
    "toolbar.field_of_view":            {"en": "Field of View",             "zh": "视野角度"},
    "toolbar.navigation_header":        {"en": "Navigation",                "zh": "导航"},
    "toolbar.rotation_sensitivity":     {"en": "Rotation Sensitivity",      "zh": "旋转灵敏度"},
    "toolbar.pan_speed":                {"en": "Pan Speed",                 "zh": "平移速度"},
    "toolbar.zoom_speed":               {"en": "Zoom Speed",                "zh": "缩放速度"},
    "toolbar.move_speed":               {"en": "Move Speed",                "zh": "移动速度"},
    "toolbar.speed_boost":              {"en": "Speed Boost",               "zh": "加速倍率"},
    "toolbar.reset_camera_settings":    {"en": "Reset Camera Settings",     "zh": "重置摄像机设置"},

    # ── Hierarchy ─────────────────────────────────────────────────────
    "hierarchy.create_child":           {"en": "Create Child",              "zh": "创建子对象"},
    "hierarchy.empty_object":           {"en": "Empty",                     "zh": "空对象"},
    "hierarchy.save_as_prefab":         {"en": "Save as Prefab",            "zh": "保存为预制体"},
    "hierarchy.prefab_label":           {"en": "Prefab",                    "zh": "预制体"},
    "hierarchy.select_prefab_asset":    {"en": "Select Prefab Asset",       "zh": "选择预制体资产"},
    "hierarchy.open_prefab":            {"en": "Open Prefab",               "zh": "打开预制体"},
    "hierarchy.apply_all_overrides":    {"en": "Apply All Overrides",       "zh": "应用所有覆盖"},
    "hierarchy.revert_all_overrides":   {"en": "Revert All Overrides",      "zh": "还原所有覆盖"},
    "hierarchy.unpack_prefab":          {"en": "Unpack Prefab",             "zh": "解除预制体链接"},
    "hierarchy.prefab_mode_header":     {"en": "[Prefab] {name}",           "zh": "[预制体] {name}"},
    "hierarchy.delete":                 {"en": "Delete",                    "zh": "删除"},
    "hierarchy.rename":                 {"en": "Rename",                    "zh": "重命名"},
    "hierarchy.ui_mode":                {"en": "UI Mode",                   "zh": "UI 模式"},
    "hierarchy.no_scene":               {"en": "(No Scene)",                "zh": "(无场景)"},
    "hierarchy.search_placeholder":     {"en": "Search objects...",         "zh": "搜索场景物体..."},
    "hierarchy.no_search_results":      {"en": "No objects match the search.", "zh": "没有匹配的物体。"},
    "hierarchy.create_3d_object":       {"en": "Create 3D Object",          "zh": "创建 3D 对象"},
    "hierarchy.light_menu":             {"en": "Light",                     "zh": "灯光"},
    "hierarchy.rendering_menu":         {"en": "Rendering",                 "zh": "渲染"},
    "hierarchy.ui_menu":                {"en": "UI",                        "zh": "UI"},
    "hierarchy.create_empty":           {"en": "Create Empty",              "zh": "创建空对象"},
    "hierarchy.delete_selected":        {"en": "Delete Selected",           "zh": "删除选中对象"},
    "hierarchy.primitive_cube":         {"en": "Cube",                      "zh": "立方体"},
    "hierarchy.primitive_sphere":       {"en": "Sphere",                    "zh": "球体"},
    "hierarchy.primitive_capsule":      {"en": "Capsule",                   "zh": "胶囊体"},
    "hierarchy.primitive_cylinder":     {"en": "Cylinder",                  "zh": "圆柱体"},
    "hierarchy.primitive_plane":        {"en": "Plane",                     "zh": "平面"},
    "hierarchy.light_directional":      {"en": "Directional Light",         "zh": "平行光"},
    "hierarchy.light_point":            {"en": "Point Light",               "zh": "点光源"},
    "hierarchy.light_spot":             {"en": "Spot Light",                "zh": "聚光灯"},
    "hierarchy.camera":                 {"en": "Camera",                    "zh": "相机"},
    "hierarchy.render_stack":           {"en": "RenderStack",               "zh": "RenderStack"},
    "hierarchy.ui_canvas":              {"en": "Canvas",                    "zh": "Canvas"},
    "hierarchy.ui_text":                {"en": "T Text",                    "zh": "T 文本"},
    "hierarchy.ui_button":              {"en": "Button",                    "zh": "按钮"},

    # ── Console ───────────────────────────────────────────────────────
    "console.clear":                    {"en": "Clear",                     "zh": "清除"},
    "console.collapse":                 {"en": "Collapse",                  "zh": "折叠"},
    "console.clear_on_play":            {"en": "Clear on Play",             "zh": "播放时清除"},
    "console.error_pause":              {"en": "Error Pause",               "zh": "错误时暂停"},

    # ── Scene view ────────────────────────────────────────────────────
    "scene_view.loading":               {"en": "Loading scene...",          "zh": "场景加载中..."},
    "scene_view.global":                {"en": "Global",                    "zh": "全局"},
    "scene_view.local":                 {"en": "Local",                     "zh": "局部"},
    "scene_view.tool_select":           {"en": "Select (Q)",               "zh": "选择 (Q)"},
    "scene_view.tool_move":             {"en": "Move (W)",                  "zh": "移动 (W)"},
    "scene_view.tool_rotate":           {"en": "Rotate (E)",               "zh": "旋转 (E)"},
    "scene_view.tool_scale":            {"en": "Scale (R)",                "zh": "缩放 (R)"},
    "scene_view.exit_prefab_mode":      {"en": "< Exit Prefab Mode",        "zh": "< 退出预制体模式"},

    # ── Game view ─────────────────────────────────────────────────────
    "game_view.engine_not_init":        {"en": "Engine not initialized",    "zh": "引擎未初始化"},
    "game_view.fit":                    {"en": "Fit",                       "zh": "适应"},
    "game_view.no_camera":              {"en": "No Camera",                 "zh": "没有摄像机"},
    "game_view.no_camera_detail":       {"en": "No Camera component in scene", "zh": "场景中没有 Camera 组件"},
    "game_view.create_camera_hint_1":   {"en": "Create a GameObject in the scene", "zh": "请在场景中创建一个 GameObject"},
    "game_view.create_camera_hint_2":   {"en": "and add a Camera component to enable Game View", "zh": "并添加 Camera 组件以启用 Game View"},

    # ── Project panel ─────────────────────────────────────────────────
    "project.create_menu":              {"en": "Create",                    "zh": "创建"},
    "project.create_folder":            {"en": "Folder",                    "zh": "文件夹"},
    "project.create_script":            {"en": "Script (.py)",              "zh": "脚本 (.py)"},
    "project.create_vert_shader":       {"en": "Vertex Shader (.vert)",     "zh": "顶点着色器 (.vert)"},
    "project.create_frag_shader":       {"en": "Fragment Shader (.frag)",   "zh": "片段着色器 (.frag)"},
    "project.create_material":          {"en": "Material (.mat)",           "zh": "材质 (.mat)"},
    "project.create_scene":             {"en": "Scene (.scene)",            "zh": "场景 (.scene)"},
    "project.reveal_in_explorer":       {"en": "Reveal in File Explorer",   "zh": "在资源管理器中显示"},
    "project.copy":                     {"en": "Copy           (Ctrl+C)",   "zh": "复制           (Ctrl+C)"},
    "project.cut":                      {"en": "Cut            (Ctrl+X)",   "zh": "剪切           (Ctrl+X)"},
    "project.paste":                    {"en": "Paste          (Ctrl+V)",   "zh": "粘贴           (Ctrl+V)"},
    "project.rename":                   {"en": "Rename         (F2)",       "zh": "重命名         (F2)"},
    "project.delete":                   {"en": "Delete         (Del)",      "zh": "删除           (Del)"},
    "project.delete_confirm_title":       {"en": "Delete Asset",               "zh": "删除资源"},
    "project.delete_confirm_msg":         {"en": "Are you sure you want to delete '{name}'?\nThis action cannot be undone.", "zh": "确定要删除 '{name}' 吗？\n此操作无法撤销。"},
    "project.delete_confirm_multi_msg":   {"en": "Are you sure you want to delete {count} selected items?\nThis action cannot be undone.", "zh": "确定要删除选中的 {count} 个项目吗？\n此操作无法撤销。"},
    "project.no_project_path":          {"en": "No project path set",       "zh": "未设置项目路径"},
    "project.empty_folder":             {"en": "(Empty folder)",            "zh": "(空文件夹)"},
    "project.right_click_hint":         {"en": "Right-click to create new items", "zh": "右键创建新项目"},
    "project.invalid_path":             {"en": "Invalid path",              "zh": "无效路径"},

    # ── Inspector ─────────────────────────────────────────────────────
    "inspector.no_selection":           {"en": "No selection",              "zh": "未选中"},
    "inspector.select_file_hint":       {"en": "Select a file in the Project panel to preview", "zh": "在项目面板中选择文件以预览"},
    "inspector.folder_label":           {"en": "Folder: {name}",            "zh": "文件夹: {name}"},
    "inspector.path_label":             {"en": "Path: {path}",              "zh": "路径: {path}"},
    "inspector.file_label":             {"en": "File: {name}",              "zh": "文件: {name}"},
    "inspector.preview_not_init":       {"en": "(Preview system not initialized)", "zh": "(预览系统未初始化)"},
    "inspector.no_previewer":           {"en": "(No previewer available for this file type)", "zh": "(此文件类型无可用预览器)"},
    "inspector.extension_label":        {"en": "Extension: {ext}",          "zh": "扩展名: {ext}"},
    "inspector.preview_failed":         {"en": "(Failed to load preview)",  "zh": "(加载预览失败)"},
    "inspector.tag":                    {"en": "Tag",                       "zh": "标签"},
    "inspector.layer":                  {"en": "Layer",                     "zh": "层"},
    "inspector.prefab_label":           {"en": "Prefab",                    "zh": "预制体"},
    "inspector.prefab_select":          {"en": "Select",                    "zh": "选择"},
    "inspector.prefab_open":            {"en": "Open",                      "zh": "打开"},
    "inspector.prefab_apply":           {"en": "Apply",                     "zh": "应用"},
    "inspector.prefab_revert":          {"en": "Revert",                    "zh": "还原"},
    "inspector.overrides":              {"en": "Override(s)",               "zh": "覆盖项"},
    "inspector.no_overrides":           {"en": "No Overrides",              "zh": "无覆盖"},
    "inspector.mesh":                   {"en": "Mesh",                      "zh": "网格"},
    "inspector.materials":              {"en": "Materials",                  "zh": "材质"},
    "inspector.material_overrides":     {"en": "Material Overrides",        "zh": "材质覆盖"},
    "inspector.no_properties":          {"en": "(No properties)",           "zh": "(无属性)"},
    "inspector.search_components":      {"en": "Search components...",      "zh": "搜索组件..."},
    "inspector.miscellaneous":          {"en": "Miscellaneous",             "zh": "其他"},
    "inspector.scripts":                {"en": "Scripts",                   "zh": "脚本"},
    "inspector.no_components_found":    {"en": "No components found",       "zh": "未找到组件"},
    "inspector.copy_properties":        {"en": "Copy Properties",           "zh": "复制属性"},
    "inspector.paste_as_new":           {"en": "Paste as New Component",    "zh": "粘贴为新组件"},
    "inspector.paste_properties":       {"en": "Paste as Properties",       "zh": "粘贴属性"},
    "inspector.remove":                 {"en": "Remove",                    "zh": "移除"},
    "inspector.show_script":            {"en": "Show Script",               "zh": "显示脚本"},
    "inspector.add_component":          {"en": "Add Component",             "zh": "添加组件"},
    "inspector.no_objects_selected":    {"en": "No objects selected",       "zh": "未选中任何对象"},
    "inspector.no_object_selected":     {"en": "No object selected",        "zh": "无对象选中"},
    "inspector.unknown_type":           {"en": "(unknown type)",            "zh": "(未知类型)"},

    # ── Build settings body ───────────────────────────────────────────
    "build.game_name":                  {"en": "Game Name",                 "zh": "游戏名称"},
    "build.game_name_hint":             {"en": "(default: {name})",         "zh": "(默认: {name})"},
    "build.debug_mode":                 {"en": "Debug Mode",               "zh": "调试模式"},
    "build.lto":                        {"en": "LTO (slower build, better perf)", "zh": "LTO（编译更慢，性能更好）"},
    "build.enable_jit":                 {"en": "JIT (Numba acceleration)",   "zh": "JIT（Numba 加速）"},
    "build.output_directory":           {"en": "Output Directory",          "zh": "输出目录"},
    "build.output_directory_hint":      {"en": "Use an empty folder, or reuse one containing {marker} from a previous Infernux build.", "zh": "请使用空文件夹，或复用包含旧 Infernux 构建标记文件 {marker} 的目录。"},
    "build.output_directory_error_title": {"en": "Invalid Build Output Directory", "zh": "无效的构建输出目录"},
    "build.output_directory_error_required": {"en": "Please choose an output directory before building.", "zh": "请先选择输出目录，再开始构建。"},
    "build.output_directory_error_path_is_file": {"en": "The selected build output path is a file, not a folder.\n\nPath: {path}", "zh": "当前选择的构建输出路径是文件，不是文件夹。\n\n路径: {path}"},
    "build.output_directory_error_not_directory": {"en": "The selected build output path is not a valid folder.\n\nPath: {path}", "zh": "当前选择的构建输出路径不是有效文件夹。\n\n路径: {path}"},
    "build.output_directory_error_not_empty": {"en": "You cannot build into this folder directly.\n\nChoose an empty folder, or reuse a folder that already contains {marker} from a previous Infernux build.\n\nFolder: {path}", "zh": "不能直接构建到这个文件夹。\n\n请选择空文件夹，或者复用一个已经包含旧 Infernux 构建标记 {marker} 的目录。\n\n目录: {path}"},
    "build.output_directory_error_found": {"en": "Existing entries: {entries}", "zh": "当前已有内容: {entries}"},
    "build.icon":                       {"en": "Build Icon",                "zh": "打包图标"},
    "build.icon_hint":                  {"en": "Leave empty to use the default engine icon", "zh": "留空则使用默认引擎图标"},
    "build.browse":                     {"en": "Browse...",                 "zh": "浏览..."},
    "build.clear_icon":                 {"en": "Clear",                     "zh": "清除"},
    "build.display_mode":               {"en": "Display Mode",              "zh": "显示模式"},
    "build.window_size":                {"en": "Window Size",               "zh": "窗口大小"},
    "build.window_resizable":           {"en": "Resizable",                 "zh": "允许调整窗口大小"},
    "build.width":                      {"en": "W",                         "zh": "宽"},
    "build.height":                     {"en": "H",                         "zh": "高"},
    "build.splash_sequence":            {"en": "Splash Sequence",           "zh": "开场画面"},
    "build.add_splash":                 {"en": "Add Image/Video",           "zh": "添加图片/视频"},
    "build.splash_help":                {"en": "Images use display time plus fade-in/out. Videos play their own length; fade values still control transitions.", "zh": "图片项会使用显示时长与淡入淡出；视频项按自身长度播放，淡入淡出仍控制切换过渡。"},
    "build.splash_file_hint_image":     {"en": "Image splash: set how long it stays on screen.", "zh": "图片开场：设置它在屏幕上停留多久。"},
    "build.splash_file_hint_video":     {"en": "Video splash: playback length comes from the video file.", "zh": "视频开场：播放时长由视频文件本身决定。"},
    "build.duration":                   {"en": "Duration",                  "zh": "时长"},
    "build.duration_hint":              {"en": "Visible hold time for this image, excluding fade-in and fade-out.", "zh": "这张图片保持可见的停留时间，不包含淡入和淡出。"},
    "build.fade_in":                    {"en": "Fade In",                   "zh": "淡入"},
    "build.fade_in_hint":               {"en": "How long the splash takes to fade from black into view.", "zh": "开场画面从黑场淡入显示所需的时间。"},
    "build.fade_out":                   {"en": "Fade Out",                  "zh": "淡出"},
    "build.fade_out_hint":              {"en": "How long the splash takes to fade back out before the next item.", "zh": "切到下一项前，当前开场画面淡出的时间。"},
    "build.seconds_short":              {"en": "sec",                       "zh": "秒"},
    "build.move_up":                    {"en": "Up",                        "zh": "上移"},
    "build.move_down":                  {"en": "Down",                      "zh": "下移"},
    "build.remove":                     {"en": "Remove",                    "zh": "删除"},
    "build.no_splash_items":            {"en": "(No splash items)",         "zh": "(无开场画面)"},
    "build.scenes_in_build":            {"en": "Scenes in Build",           "zh": "构建场景列表"},
    "build.add_open_scene":             {"en": "Add Open Scene",            "zh": "添加当前场景"},
    "build.list_empty":                 {"en": "(Build list is empty)",     "zh": "(构建列表为空)"},
    "build.drag_scenes_hint":           {"en": "Drag scenes from the Project panel here.", "zh": "将场景从项目面板拖入此处。"},
    "build.building":                   {"en": "Building...",               "zh": "构建中..."},
    "build.cancel":                     {"en": "Cancel",                    "zh": "取消"},
    "build.cancelled":                  {"en": "Build cancelled.",          "zh": "构建已取消。"},
    "build.failed":                     {"en": "Build failed: {err}",       "zh": "构建失败: {err}"},
    "build.succeeded":                  {"en": "Build succeeded → {path}",  "zh": "构建成功 → {path}"},
    "build.completed_log":              {"en": "Build completed in {seconds:.2f}s: {path}", "zh": "构建完成，用时 {seconds:.2f} 秒：{path}"},
    "build.open_folder":                {"en": "Open Folder",               "zh": "打开文件夹"},
    "build.build":                      {"en": "Build",                     "zh": "构建"},
    "build.build_and_run":              {"en": "Build And Run",             "zh": "构建并运行"},

    # ── Build pipeline steps ──────────────────────────────────────────
    "build.step.validating":            {"en": "Validating project...",                 "zh": "验证项目..."},
    "build.step.cleaning_output":       {"en": "Cleaning output...",                    "zh": "清理输出目录..."},
    "build.step.collecting_deps":       {"en": "Collecting user dependencies...",       "zh": "收集用户依赖..."},
    "build.step.generating_boot":       {"en": "Generating boot script...",             "zh": "生成入口脚本..."},
    "build.step.nuitka_compilation":    {"en": "Nuitka native compilation...",          "zh": "Nuitka 原生编译..."},
    "build.step.organizing_output":     {"en": "Organizing output directory...",        "zh": "整理输出目录..."},
    "build.step.copying_data":          {"en": "Copying game data...",                  "zh": "复制游戏数据..."},
    "build.step.compiling_scripts":     {"en": "Compiling user scripts...",             "zh": "编译用户脚本..."},
    "build.step.processing_splash":     {"en": "Processing splash items...",            "zh": "处理开场画面..."},
    "build.step.fixing_scenes":         {"en": "Fixing scene paths...",                 "zh": "处理场景路径..."},
    "build.step.generating_manifest":   {"en": "Generating manifest...",                "zh": "生成构建清单..."},
    "build.step.cleaning_redundant":    {"en": "Cleaning redundant resources...",       "zh": "清理冗余资源..."},
    "build.step.writing_marker":        {"en": "Writing build marker...",               "zh": "写入构建标记..."},
    "build.step.cleaning_temp":         {"en": "Cleaning temp files...",                "zh": "清理临时文件..."},
    "build.step.complete":              {"en": "Build complete!",                       "zh": "构建完成！"},
    # Nuitka sub-steps
    "build.step.checking_nuitka":       {"en": "Checking Nuitka...",                    "zh": "检查 Nuitka 可用性..."},
    "build.step.preparing_staging":     {"en": "Preparing staging directory...",        "zh": "准备暂存目录..."},
    "build.step.building_command":      {"en": "Building command...",                   "zh": "构建 Nuitka 命令..."},
    "build.step.running_nuitka":        {"en": "Running Nuitka compilation...",         "zh": "执行 Nuitka 编译..."},
    "build.step.injecting_libs":        {"en": "Injecting native engine libraries...",  "zh": "注入原生引擎库..."},
    "build.step.injecting_jit":         {"en": "Injecting JIT runtime packages...",     "zh": "注入 JIT 运行时包..."},
    "build.step.embedding_manifest":    {"en": "Embedding UTF-8 manifest...",           "zh": "嵌入 UTF-8 清单..."},
    "build.step.signing_exe":           {"en": "Signing executable...",                 "zh": "签名可执行文件..."},
    "build.step.cleaning_artifacts":    {"en": "Cleaning build artifacts...",           "zh": "清理编译产物..."},
    "build.step.nuitka_complete":       {"en": "Compilation complete!",                 "zh": "Nuitka 编译完成！"},

    # ── Physics / Tags & Layers ───────────────────────────────────────
    "physics.title":                    {"en": "Physics Settings",          "zh": "物理设置"},
    "physics.collision_matrix_hint":    {"en": "Collision Matrix — upper triangle only. Changes are saved immediately.", "zh": "碰撞矩阵 — 仅上三角。更改立即保存。"},
    "physics.no_layers":                {"en": "No layers available",       "zh": "没有可用图层"},
    "physics.simulation":               {"en": "Simulation",                "zh": "模拟"},
    "physics.iteration_rate":           {"en": "Physics Iteration Rate (Hz)", "zh": "物理迭代频率 (Hz)"},
    "physics.fixed_time_step":          {"en": "Fixed Time Step (s)",       "zh": "固定时间步长 (s)"},
    "physics.max_catchup_delta":        {"en": "Max Catch-up Delta (s)",    "zh": "最大追赶间隔 (s)"},
    "physics.gravity":                  {"en": "Gravity",                   "zh": "重力"},
    "tags.manager_unavailable":         {"en": "TagLayerManager not available", "zh": "TagLayerManager 不可用"},
    "tags.tags_header":                 {"en": "Tags",                      "zh": "标签"},
    "tags.built_in":                    {"en": "(built-in)",                "zh": "(内置)"},
    "tags.add_tag":                     {"en": "Add Tag:",                  "zh": "添加标签:"},
    "tags.layers_header":               {"en": "Layers",                    "zh": "图层"},
    "tags.save_settings":               {"en": "Save Settings",             "zh": "保存设置"},
    "tags.reset_defaults":              {"en": "Reset to Defaults",         "zh": "重置为默认"},

    # ── UI Editor ─────────────────────────────────────────────────────
    "ui_editor.no_canvas":              {"en": "No UICanvas in scene",      "zh": "场景中没有 Canvas"},
    "ui_editor.create_canvas_hint":     {"en": "Right-click in Hierarchy → Create → UI → Canvas\nor use the button below", "zh": "在 Hierarchy 中右键 → 创建 → UI → Canvas\n或使用下方按钮快速创建"},
    "ui_editor.create_canvas":          {"en": "Create Canvas",             "zh": "创建 Canvas"},
    "ui_editor.tooltip_canvas":         {"en": "Create Canvas",              "zh": "创建 Canvas"},
    "ui_editor.tooltip_text":           {"en": "Text",                      "zh": "文本"},
    "ui_editor.tooltip_image":          {"en": "Image",                     "zh": "图像"},
    "ui_editor.tooltip_button":         {"en": "Button",                    "zh": "按钮"},
    "ui_editor.zoom":                   {"en": "Zoom: {pct}%",              "zh": "缩放: {pct}%"},
    "ui_editor.fit":                    {"en": "Fit",                       "zh": "适应"},

    # ── Material inspector ────────────────────────────────────────────
    "material.builtin_locked":          {"en": "(Built-in — Shader locked)", "zh": "(内置 — 着色器锁定)"},
    "material.shader_section":          {"en": "Shader",                    "zh": "着色器"},
    "material.vertex":                  {"en": "Vertex",                    "zh": "顶点"},
    "material.fragment":                {"en": "Fragment",                   "zh": "片段"},
    "material.surface_options":         {"en": "Surface Options",           "zh": "表面选项"},
    "material.surface_type":            {"en": "Surface Type",              "zh": "表面类型"},
    "material.cull_mode":               {"en": "Cull Mode",                 "zh": "剔除模式"},
    "material.depth_write":             {"en": "Depth Write",               "zh": "深度写入"},
    "material.depth_test":              {"en": "Depth Test",                "zh": "深度测试"},
    "material.blend_mode":              {"en": "Blend Mode",                "zh": "混合模式"},
    "material.alpha_clip":              {"en": "Alpha Clip",                "zh": "Alpha 裁剪"},
    "material.render_queue":            {"en": "Render Queue",              "zh": "渲染队列"},
    "material.opaque":                  {"en": "Opaque",                    "zh": "不透明"},
    "material.transparent":             {"en": "Transparent",               "zh": "透明"},
    "material.cull_none":               {"en": "None",                      "zh": "无"},
    "material.cull_front":              {"en": "Front",                     "zh": "正面"},
    "material.cull_back":               {"en": "Back",                      "zh": "背面"},
    "material.compare_never":           {"en": "Never",                     "zh": "从不"},
    "material.compare_less":            {"en": "Less",                      "zh": "小于"},
    "material.compare_equal":           {"en": "Equal",                     "zh": "等于"},
    "material.compare_less_equal":      {"en": "Less or Equal",             "zh": "小于等于"},
    "material.compare_greater":         {"en": "Greater",                   "zh": "大于"},
    "material.compare_not_equal":       {"en": "Not Equal",                 "zh": "不等于"},
    "material.compare_greater_equal":   {"en": "Greater or Equal",          "zh": "大于等于"},
    "material.compare_always":          {"en": "Always",                    "zh": "始终"},
    "material.blend_alpha":             {"en": "Alpha",                     "zh": "Alpha"},
    "material.blend_additive":          {"en": "Additive",                  "zh": "叠加"},
    "material.blend_premultiply":       {"en": "Premultiply",               "zh": "预乘"},
    "material.threshold":               {"en": "Threshold",                 "zh": "阈值"},
    "material.no_properties":           {"en": "(No properties)",           "zh": "(无属性)"},
    "material.properties_section":      {"en": "Properties",                "zh": "属性"},

    # ── Render stack inspector ────────────────────────────────────────
    "renderstack.pipeline":             {"en": "Pipeline",                  "zh": "管线"},
    "renderstack.default_forward":      {"en": "Default Forward",           "zh": "默认前向"},
    "renderstack.pipeline_settings":    {"en": "Pipeline Settings",         "zh": "管线设置"},
    "renderstack.empty_topology":       {"en": "(empty topology)",          "zh": "(空拓扑)"},
    "renderstack.add_pass":             {"en": "Add Pass...",               "zh": "添加 Pass..."},
    "renderstack.no_passes":            {"en": "No passes available",       "zh": "无可用 Pass"},
    "renderstack.post_processing":      {"en": "Post-processing",           "zh": "后处理"},
    "renderstack.geometry":             {"en": "Geometry",                   "zh": "几何"},
    "renderstack.other":                {"en": "Other",                     "zh": "其他"},
    "renderstack.remove":               {"en": "Remove",                    "zh": "移除"},

    # ── UI component inspector ────────────────────────────────────────
    "ui_comp.location":                 {"en": "Location",                  "zh": "位置"},
    "ui_comp.alignment":                {"en": "Alignment",                 "zh": "对齐"},
    "ui_comp.position":                 {"en": "Position",                  "zh": "坐标"},
    "ui_comp.no_canvas_context":        {"en": "No canvas context",         "zh": "无画布上下文"},
    "ui_comp.rotation":                 {"en": "Rotation",                  "zh": "旋转"},
    "ui_comp.rotate_90":                {"en": "+90",                       "zh": "+90"},
    "ui_comp.mirror_h":                 {"en": "Mirror H",                  "zh": "水平镜像"},
    "ui_comp.mirror_v":                 {"en": "Mirror V",                  "zh": "垂直镜像"},
    "ui_comp.layout":                   {"en": "Layout",                    "zh": "布局"},
    "ui_comp.dimensions":               {"en": "Dimensions",                "zh": "尺寸"},
    "ui_comp.lock":                     {"en": "Lock",                      "zh": "锁定"},
    "ui_comp.set_native_size":          {"en": "Set Native Size",           "zh": "设为原始大小"},
    "ui_comp.resizing":                 {"en": "Resizing",                  "zh": "调整大小"},
    "ui_comp.auto_width":               {"en": "Auto Width",                "zh": "自动宽度"},
    "ui_comp.auto_height":              {"en": "Auto Height",               "zh": "自动高度"},
    "ui_comp.fixed_size":               {"en": "Fixed Size",                "zh": "固定大小"},
    "ui_comp.appearance":               {"en": "Appearance",                "zh": "外观"},
    "ui_comp.opacity":                  {"en": "Opacity",                   "zh": "不透明度"},
    "ui_comp.corner_radius":            {"en": "Corner Radius",             "zh": "圆角半径"},
    "ui_comp.typography":               {"en": "Typography",                "zh": "排版"},
    "ui_comp.content":                  {"en": "Content",                   "zh": "内容"},
    "ui_comp.font":                     {"en": "Font",                      "zh": "字体"},
    "ui_comp.font_size":                {"en": "Font Size",                 "zh": "字号"},
    "ui_comp.line_height":              {"en": "Line Height",               "zh": "行高"},
    "ui_comp.letter_spacing":           {"en": "Letter Spacing",            "zh": "字间距"},
    "ui_comp.text_alignment":           {"en": "Text Alignment",            "zh": "文本对齐"},
    "ui_comp.fill":                     {"en": "Fill",                      "zh": "填充"},
    "ui_comp.color":                    {"en": "Color",                     "zh": "颜色"},
    "ui_comp.canvas":                   {"en": "Canvas",                    "zh": "画布"},
    "ui_comp.render_mode":              {"en": "Render Mode",               "zh": "渲染模式"},
    "ui_comp.sort_order":               {"en": "Sort Order",                "zh": "排序顺序"},
    "ui_comp.target_camera":            {"en": "Target Camera",             "zh": "目标摄像机"},
    "ui_comp.reference_size":           {"en": "Reference Size",            "zh": "参考大小"},
    "ui_comp.canvas_scaler":            {"en": "Canvas Scaler",             "zh": "画布缩放器"},
    "ui_comp.ui_scale_mode":            {"en": "UI Scale Mode",             "zh": "UI缩放模式"},
    "ui_comp.screen_match_mode":        {"en": "Screen Match Mode",         "zh": "屏幕匹配模式"},
    "ui_comp.match":                    {"en": "Match",                     "zh": "匹配"},
    "ui_comp.pixel_perfect":            {"en": "Pixel Perfect",             "zh": "像素完美"},
    "ui_comp.ref_pixels_per_unit":      {"en": "Reference Pixels Per Unit", "zh": "参考每单位像素"},
    "ui_comp.texture":                  {"en": "Texture",                   "zh": "贴图"},
    "ui_comp.interaction":              {"en": "Interaction",               "zh": "交互"},
    "ui_comp.interactable":             {"en": "Interactable",              "zh": "可交互"},
    "ui_comp.raycast_target":           {"en": "Raycast Target",            "zh": "射线检测目标"},
    "ui_comp.label":                    {"en": "Label",                     "zh": "标签文字"},
    "ui_comp.label_color":              {"en": "Label Color",               "zh": "标签颜色"},
    "ui_comp.background":               {"en": "Background",                "zh": "背景"},
    "ui_comp.color_tint":               {"en": "ColorTint",                 "zh": "颜色着色"},
    "ui_comp.tint_normal":              {"en": "Normal",                    "zh": "正常"},
    "ui_comp.tint_highlighted":         {"en": "Highlighted",               "zh": "高亮"},
    "ui_comp.tint_pressed":             {"en": "Pressed",                   "zh": "按下"},
    "ui_comp.tint_disabled":            {"en": "Disabled",                  "zh": "禁用"},
    "ui_comp.no_parameters":            {"en": "No parameters",             "zh": "无参数"},
    "ui_comp.default_font":             {"en": "Default",                   "zh": "默认"},
    "ui_comp.align_left":               {"en": "Left",                      "zh": "左"},
    "ui_comp.align_cx":                 {"en": "CX",                        "zh": "中X"},
    "ui_comp.align_right":              {"en": "Right",                     "zh": "右"},
    "ui_comp.align_top":                {"en": "Top",                       "zh": "上"},
    "ui_comp.align_mid":                {"en": "Mid",                       "zh": "中"},
    "ui_comp.align_bot":                {"en": "Bot",                       "zh": "下"},
    "ui_comp.modify":                   {"en": "Modify",                    "zh": "修改"},
    "ui_comp.size":                     {"en": "Size",                      "zh": "大小"},
    "ui_comp.text_left":                {"en": "Left",                      "zh": "左对齐"},
    "ui_comp.text_center":              {"en": "Center",                    "zh": "居中"},
    "ui_comp.text_right":               {"en": "Right",                     "zh": "右对齐"},
    "ui_comp.text_top":                 {"en": "Top",                       "zh": "顶部"},
    "ui_comp.text_middle":              {"en": "Middle",                    "zh": "中间"},
    "ui_comp.text_bottom":              {"en": "Bottom",                    "zh": "底部"},
    "ui_comp.on_click":                 {"en": "On Click",                  "zh": "点击事件"},
    "ui_comp.click_entry":              {"en": "Click {n}",                 "zh": "点击 {n}"},
    "ui_comp.target":                   {"en": "Target",                    "zh": "目标"},
    "ui_comp.component":                {"en": "Component",                 "zh": "组件"},
    "ui_comp.method":                   {"en": "Method",                    "zh": "方法"},
    "ui_comp.arguments":                {"en": "Arguments",                 "zh": "参数"},
    "ui_comp.none":                     {"en": "(None)",                    "zh": "(无)"},
    "ui_comp.params_count":             {"en": "{n} parameter(s)",          "zh": "{n} 个参数"},

    # ── Asset inspector ───────────────────────────────────────────────
    "asset.display_texture":            {"en": "Texture",                   "zh": "贴图"},
    "asset.texture_type":               {"en": "Texture Type",              "zh": "贴图类型"},
    "asset.srgb":                       {"en": "sRGB",                      "zh": "sRGB"},
    "asset.max_size":                   {"en": "Max Size",                  "zh": "最大尺寸"},
    "asset.tex_default":                {"en": "Default",                   "zh": "默认"},
    "asset.tex_normalmap":              {"en": "NormalMap",                  "zh": "法线贴图"},
    "asset.tex_ui":                     {"en": "UI",                        "zh": "UI"},
    "asset.display_audio":              {"en": "Audio",                     "zh": "音频"},
    "asset.force_mono":                 {"en": "Force Mono",                "zh": "强制单声道"},
    "asset.display_shader":             {"en": "Shader",                    "zh": "着色器"},
    "asset.display_font":               {"en": "Font",                      "zh": "字体"},
    "asset.display_mesh":               {"en": "Mesh",                      "zh": "网格"},
    "asset.scale_factor":               {"en": "Scale Factor",              "zh": "缩放因子"},
    "asset.generate_normals":           {"en": "Generate Normals",          "zh": "生成法线"},
    "asset.generate_tangents":          {"en": "Generate Tangents",         "zh": "生成切线"},
    "asset.flip_uvs":                   {"en": "Flip UVs",                  "zh": "翻转 UV"},
    "asset.optimize_mesh":              {"en": "Optimize Mesh",             "zh": "优化网格"},
    "asset.display_material":           {"en": "Material",                  "zh": "材质"},
    "asset.display_prefab":             {"en": "Prefab",                    "zh": "预制体"},
    "asset.invalid_prefab":             {"en": "Invalid prefab data",       "zh": "无效的预制体数据"},
    "asset.open_prefab_mode":           {"en": "Open Prefab Mode",          "zh": "打开预制体模式"},
    "asset.prefab_safe_mode":           {"en": "Inline prefab inspector is temporarily running in safe mode to avoid editor crashes.", "zh": "内联预制体检视器当前以安全模式运行，以避免编辑器崩溃。"},
    "asset.prefab_root":                {"en": "Root",                      "zh": "根节点"},
    "asset.prefab_nodes":               {"en": "Nodes",                     "zh": "节点数"},
    "asset.prefab_components_count":    {"en": "Components",                "zh": "组件数"},
    "asset.prefab_scripts_count":       {"en": "Scripts",                   "zh": "脚本数"},
    "asset.prefab_path":                {"en": "Path",                      "zh": "路径"},
    "asset.prefab_root_object":         {"en": "Root Object",               "zh": "根对象"},
    "asset.prefab_raw_json_preview":    {"en": "Raw JSON Preview",          "zh": "原始 JSON 预览"},
    "asset.prefab_native_components":   {"en": "Native Components",         "zh": "原生组件"},
    "asset.prefab_script_components":   {"en": "Script Components",         "zh": "脚本组件"},
    "asset.prefab_children_count":      {"en": "Children",                  "zh": "子节点"},
    "asset.prefab_children":            {"en": "Children ({n})",            "zh": "子节点 ({n})"},
    "asset.prefab_summary":             {"en": "Total Nodes: {nodes} | Root Components: {comps}", "zh": "总节点: {nodes} | 根组件: {comps}"},
    "asset.no_editable_fields":         {"en": "(no editable fields)",      "zh": "(无可编辑字段)"},
    "asset.file_not_exist_warning":     {"en": "Warning: file does not exist", "zh": "警告: 文件不存在"},
    "asset.apply_path_change":          {"en": "Apply Path Change",         "zh": "应用路径更改"},
    "asset.file_not_found":             {"en": "(file not found)",          "zh": "(文件未找到)"},
    "asset.failed_read_source":         {"en": "(failed to read source)",   "zh": "(读取源文件失败)"},
    "asset.import_settings":            {"en": "Import Settings",           "zh": "导入设置"},
    "asset.unknown_asset_type":         {"en": "Unknown asset type: {cat}", "zh": "未知资产类型: {cat}"},
    "asset.failed_load":                {"en": "Failed to load {name}",     "zh": "加载 {name} 失败"},
    "asset.mesh_info":                  {"en": "Mesh Info",                 "zh": "网格信息"},
    "asset.mesh_file":                  {"en": "File",                      "zh": "文件"},
    "asset.mesh_meshes":                {"en": "Meshes",                    "zh": "网格数"},
    "asset.mesh_vertices":              {"en": "Vertices",                  "zh": "顶点数"},
    "asset.mesh_indices":               {"en": "Indices",                   "zh": "索引数"},
    "asset.mesh_material_slots":        {"en": "Material Slots",            "zh": "材质插槽"},
    "asset.mesh_materials":             {"en": "Materials",                 "zh": "材质"},
    "asset.shader_type":                {"en": "Type",                      "zh": "类型"},
    "asset.shader_unknown":             {"en": "Unknown",                   "zh": "未知"},
    "asset.shader_path":                {"en": "Path",                      "zh": "路径"},
    "asset.shader_source_path":         {"en": "Source Path",               "zh": "源路径"},
    "asset.shader_invalid_ext":         {"en": "Invalid shader extension: {ext}", "zh": "无效着色器扩展名: {ext}"},
    "asset.shader_source_preview":      {"en": "Source Preview",            "zh": "源代码预览"},
    "asset.shader_truncated":           {"en": "... (truncated)",           "zh": "... (已截断)"},
    "asset.font_format":                {"en": "Format",                    "zh": "格式"},
    "asset.font_source_path":           {"en": "Source Path",               "zh": "源路径"},
    "asset.font_unknown":               {"en": "Unknown",                   "zh": "未知"},
    "asset.prefab_name":                {"en": "Name",                      "zh": "名称"},
    "asset.prefab_active":              {"en": "Active",                    "zh": "启用"},
    "asset.prefab_tag":                 {"en": "Tag",                       "zh": "标签"},
    "asset.prefab_layer":               {"en": "Layer",                     "zh": "层"},
    "asset.prefab_transform":           {"en": "Transform",                 "zh": "变换"},
    "asset.prefab_script":              {"en": " (Script)",                 "zh": " (脚本)"},
    "asset.prefab_position":            {"en": "Position",                  "zh": "位置"},
    "asset.prefab_rotation":            {"en": "Rotation",                  "zh": "旋转"},
    "asset.prefab_scale":               {"en": "Scale",                     "zh": "缩放"},
    "asset.guid_label":                 {"en": "GUID: {guid}",              "zh": "GUID: {guid}"},
    "asset.path_label":                 {"en": "Path: {path}",              "zh": "路径: {path}"},
    "asset.size_mb":                    {"en": "Size: {size} MB",           "zh": "大小: {size} MB"},
    "asset.size_kb":                    {"en": "Size: {size} KB",           "zh": "大小: {size} KB"},
    "asset.size_bytes":                 {"en": "Size: {size} bytes",        "zh": "大小: {size} 字节"},

    # ── iGUI helpers ──────────────────────────────────────────────────
    "igui.search_hint":                 {"en": "Search...",                 "zh": "搜索..."},
    "igui.none":                        {"en": "None",                      "zh": "无"},
    "igui.tab_scene":                   {"en": "Scene",                     "zh": "场景"},
    "igui.tab_assets":                  {"en": "Assets",                    "zh": "资产"},
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def t(key: str) -> str:
    """Return the translated string for *key* in the current locale.

    Falls back to English, then returns the key itself if not found.
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(_current_locale, entry.get("en", key))


def get_locale() -> str:
    """Return the current locale code (``"en"`` or ``"zh"``)."""
    return _current_locale


def set_locale(locale: str) -> None:
    """Set the active locale and persist to disk."""
    global _current_locale
    if locale not in ("en", "zh"):
        return
    _current_locale = locale
    _save_preference()


# ---------------------------------------------------------------------------
# Persistence — Documents/Infernux/preferences.json
# ---------------------------------------------------------------------------

_PREFS_FILE = "preferences.json"


def _prefs_path() -> str:
    """Return the path to the global preferences file."""
    if os.name == "nt":
        docs = pathlib.Path.home() / "Documents"
        try:
            import ctypes.wintypes
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
            if buf.value:
                docs = pathlib.Path(buf.value)
        except (OSError, ValueError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    else:
        docs = pathlib.Path.home() / "Documents"
    prefs_dir = docs / "Infernux"
    os.makedirs(prefs_dir, exist_ok=True)
    return str(prefs_dir / _PREFS_FILE)


def _load_preference() -> None:
    """Load the locale from the preferences file."""
    global _current_locale
    path = _prefs_path()
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        locale = data.get("language", "zh")
        if locale in ("en", "zh"):
            _current_locale = locale
    except (json.JSONDecodeError, OSError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


def _save_preference() -> None:
    """Save the current locale to the preferences file."""
    path = _prefs_path()
    data: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data["language"] = _current_locale
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


# ---------------------------------------------------------------------------
# Module init — load persisted locale on import
# ---------------------------------------------------------------------------

_load_preference()
