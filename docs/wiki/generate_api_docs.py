#!/usr/bin/env python3
"""
InfEngine API Reference — Automated Documentation Generator
============================================================

Introspects the InfEngine Python package (via .pyi stubs and source files)
and generates per-class / per-module Markdown pages in the Unity Scripting
API Reference style.

Usage:
    python docs/wiki/generate_api_docs.py

Output:
    docs/wiki/docs/en/api/*.md   (English)
    docs/wiki/docs/zh/api/*.md   (Chinese)
    docs/wiki/mkdocs_api_nav.yml (YAML fragment to paste into mkdocs.yml nav)

Design:
    • Each public class/function/enum gets its own page
    • Pages list Properties, Methods, Static Methods, Enums, Constructors
    • Description / Example sections are left as placeholders for manual editing
    • A "namespace → class" hierarchy mirrors the Python package layout
    • Bi-lingual: generates both en/ and zh/ pages

Merge strategy:
    If a generated .md file already has user-written content inside
    <!-- USER CONTENT START --> / <!-- USER CONTENT END --> markers,
    the generator preserves that content on re-generation.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WIKI_ROOT = Path(__file__).resolve().parent          # docs/wiki
DOCS_ROOT = WIKI_ROOT / "docs"
PROJECT_ROOT = WIKI_ROOT.parent.parent               # InfEngine repo root
PYTHON_ROOT = PROJECT_ROOT / "python"
STUB_ROOT = PYTHON_ROOT / "InfEngine"

EN_API = DOCS_ROOT / "en" / "api"
ZH_API = DOCS_ROOT / "zh" / "api"

# Marker for user-editable content blocks
USER_START = "<!-- USER CONTENT START -->"
USER_END   = "<!-- USER CONTENT END -->"

# Languages
LANG_EN = "en"
LANG_ZH = "zh"

# Translation table for section headings
I18N = {
    "description":          {"en": "Description",           "zh": "描述"},
    "properties":           {"en": "Properties",            "zh": "属性"},
    "constructors":         {"en": "Constructors",          "zh": "构造函数"},
    "public_methods":       {"en": "Public Methods",        "zh": "公共方法"},
    "static_methods":       {"en": "Static Methods",        "zh": "静态方法"},
    "operators":            {"en": "Operators",             "zh": "运算符"},
    "enums":                {"en": "Enums",                 "zh": "枚举"},
    "values":               {"en": "Values",                "zh": "枚举值"},
    "example":              {"en": "Example",               "zh": "示例"},
    "see_also":             {"en": "See Also",              "zh": "另请参阅"},
    "class_in":             {"en": "class in",              "zh": "类位于"},
    "enum_in":              {"en": "enum in",               "zh": "枚举位于"},
    "function_in":          {"en": "function in",           "zh": "函数位于"},
    "module":               {"en": "module",                "zh": "模块"},
    "name":                 {"en": "Name",                  "zh": "名称"},
    "type":                 {"en": "Type",                  "zh": "类型"},
    "desc":                 {"en": "Description",           "zh": "描述"},
    "method":               {"en": "Method",                "zh": "方法"},
    "returns":              {"en": "Returns",               "zh": "返回值"},
    "parameters":           {"en": "Parameters",            "zh": "参数"},
    "signature":            {"en": "Signature",             "zh": "签名"},
    "inherited_from":       {"en": "Inherited from",        "zh": "继承自"},
    "inherits":             {"en": "Inherits from",         "zh": "继承自"},
    "package":              {"en": "Package",               "zh": "包"},
    "packages":             {"en": "Packages",              "zh": "包"},
    "version":              {"en": "Version 0.1",           "zh": "版本 0.1"},
    "api_ref_title":        {"en": "InfEngine Scripting API", "zh": "InfEngine 脚本 API"},
    "api_ref_welcome":      {
        "en": "Welcome to the InfEngine Scripting API Reference. Browse packages from the sidebar to see class documentation.",
        "zh": "欢迎查阅 InfEngine 脚本 API 参考文档。请从侧边栏浏览包以查看类文档。",
    },
    "decorator_in":         {"en": "decorator in",          "zh": "装饰器位于"},
    "lifecycle_methods":    {"en": "Lifecycle Methods",     "zh": "生命周期方法"},
}

# ---------------------------------------------------------------------------
# Chinese (ZH) short descriptions for properties / methods
# ---------------------------------------------------------------------------
# Key format:  ZH_SHORT["ClassName"]["member_name"] → Chinese one-liner
# Special key "_class_" → class-level description (replaces English docstring)

ZH_SHORT: Dict[str, Dict[str, str]] = {
    "GameObject": {
        "_class_": "场景层级中具有组件的游戏对象。",
        "name": "此 GameObject 的名称。",
        "active": "此 GameObject 是否处于活动状态。",
        "id": "唯一对象标识符。",
        "transform": "获取 Transform 组件。",
        "active_self": "此对象自身是否处于活动状态。active 的别名。",
        "active_in_hierarchy": "此对象在层级中是否处于活动状态。",
        "is_static": "静态标志。",
        "scene": "此 GameObject 所属的场景。",
        "tag": "此 GameObject 的标签字符串。",
        "layer": "此 GameObject 的层级索引 (0-31)。",
        "get_transform": "获取 Transform 组件。",
        "add_component": "通过类型或类型名称添加 C++ 组件。",
        "remove_component": "移除一个组件实例（无法移除 Transform）。",
        "get_components": "获取所有组件（包括 Transform）。",
        "get_cpp_component": "根据类型名称获取 C++ 组件。",
        "get_cpp_components": "获取指定类型名称的所有 C++ 组件。",
        "add_py_component": "向此 GameObject 添加 Python InfComponent 实例。",
        "get_py_component": "获取指定类型的 Python 组件。",
        "get_py_components": "获取附加到此 GameObject 的所有 Python 组件。",
        "remove_py_component": "移除一个 Python 组件实例。",
        "get_parent": "获取父级 GameObject。",
        "set_parent": "设置父级 GameObject（None 表示根级）。",
        "get_children": "获取子 GameObject 列表。",
        "get_child_count": "获取子对象数量。",
        "is_active_in_hierarchy": "检查此对象及所有父对象是否处于活动状态。",
        "get_child": "根据索引获取子对象。",
        "find_child": "根据名称查找直接子对象（非递归）。",
        "find_descendant": "根据名称查找后代对象（递归深度优先搜索）。",
        "compare_tag": "此 GameObject 的标签是否与给定标签匹配。",
        "serialize": "将 GameObject 序列化为 JSON 字符串。",
        "deserialize": "从 JSON 字符串反序列化 GameObject。",
    },
    "Transform": {
        "_class_": "对象在场景中的位置、旋转和缩放。",
        "position": "世界空间中的位置。",
        "rotation": "世界空间中的旋转（四元数）。",
        "local_position": "相对于父变换的位置。",
        "local_rotation": "相对于父变换的旋转。",
        "local_scale": "相对于父变换的缩放。",
        "euler_angles": "世界空间中以欧拉角表示的旋转（度）。",
        "local_euler_angles": "相对于父变换的欧拉角旋转（度）。",
        "forward": "世界空间中的前方向向量（蓝轴）。",
        "right": "世界空间中的右方向向量（红轴）。",
        "up": "世界空间中的上方向向量（绿轴）。",
        "lossy_scale": "对象的全局缩放（只读）。",
        "parent": "父变换。",
        "child_count": "子变换数量。",
        "root": "层级视图中最顶层的变换。",
        "translate": "按指定方向和距离移动变换。",
        "rotate": "按欧拉角旋转变换。",
        "rotate_around": "围绕指定轴和点旋转。",
        "look_at": "旋转变换使前方向指向目标位置。",
        "get_child": "按索引获取子变换。",
        "find": "按名称查找子变换。",
        "set_parent": "设置变换的父级。",
        "set_as_first_sibling": "将变换移动到兄弟列表的开头。",
        "set_as_last_sibling": "将变换移动到兄弟列表的末尾。",
        "get_sibling_index": "获取同级索引。",
        "set_sibling_index": "设置同级索引。",
        "detach_children": "清除所有子项的父级。",
        "inverse_transform_point": "将点从世界空间变换到本地空间。",
        "transform_point": "将点从本地空间变换到世界空间。",
        "inverse_transform_direction": "将方向从世界空间变换到本地空间。",
        "transform_direction": "将方向从本地空间变换到世界空间。",
    },
    "Component": {
        "_class_": "附加到 GameObject 的所有组件的基类。",
        "game_object": "此组件附加到的 GameObject。",
        "transform": "附加到此 GameObject 的 Transform。",
        "tag": "此 GameObject 的标签。",
        "name": "此 GameObject 的名称。",
        "enabled": "此组件是否已启用。",
        "get_component": "获取同一 GameObject 上指定类型的组件。",
        "get_components": "获取同一 GameObject 上指定类型的所有组件。",
        "get_component_in_children": "在子对象中查找指定类型的组件。",
        "get_component_in_parent": "在父对象中查找指定类型的组件。",
    },
    "InfComponent": {
        "_class_": "用户脚本组件的基类，类似于 Unity 的 MonoBehaviour。",
        "game_object": "此组件附加到的 GameObject。",
        "transform": "附加到此 GameObject 的 Transform。",
        "enabled": "此组件是否已启用。",
        "is_destroyed": "此组件是否已被销毁。",
        "awake": "组件创建时调用一次。",
        "start": "首次 Update 之前调用一次。",
        "update": "每帧调用一次。",
        "fixed_update": "以固定时间间隔调用。",
        "late_update": "在所有 Update 调用之后每帧调用。",
        "on_enable": "组件启用时调用。",
        "on_disable": "组件禁用时调用。",
        "on_destroy": "组件即将被销毁时调用。",
        "on_validate": "编辑器中属性变更时调用。",
        "on_draw_gizmos": "每帧绘制 Gizmos 时调用。",
        "on_draw_gizmos_selected": "选中时绘制 Gizmos。",
        "start_coroutine": "启动一个协程。",
        "stop_coroutine": "停止一个协程。",
        "stop_all_coroutines": "停止所有协程。",
        "invoke": "在指定延迟后调用方法。",
        "invoke_repeating": "以固定间隔重复调用方法。",
        "cancel_invoke": "取消所有 Invoke 调用。",
        "destroy": "销毁此组件或指定的 GameObject。",
        "instantiate": "克隆一个 GameObject。",
    },
    "Scene": {
        "_class_": "运行时场景，包含 GameObject 层级。",
        "name": "场景名称。",
        "path": "场景路径。",
        "is_loaded": "场景是否已加载。",
        "root_count": "场景中根 GameObject 的数量。",
        "get_root_game_objects": "获取场景中所有根 GameObject。",
    },
    "SceneManager": {
        "_class_": "运行时场景加载与卸载管理器。",
        "load_scene": "按名称或路径加载场景。",
        "get_active_scene": "获取当前活动场景。",
        "set_active_scene": "设置当前活动场景。",
        "get_scene_count": "获取已加载的场景数量。",
        "get_scene_at": "按索引获取已加载的场景。",
        "create_scene": "创建一个新的空场景。",
    },
    "Camera": {
        "_class_": "渲染场景视图的摄像机组件。",
        "projection_mode": "投影模式（0=透视，1=正交）。",
        "field_of_view": "垂直视野角度（度）。",
        "orthographic_size": "正交模式下摄像机的半尺寸。",
        "aspect_ratio": "摄像机宽高比（宽/高）。",
        "near_clip": "近裁剪面距离。",
        "far_clip": "远裁剪面距离。",
        "depth": "摄像机渲染顺序。",
        "culling_mask": "用于剔除对象的图层遮罩。",
        "clear_flags": "摄像机渲染前清除背景的方式。",
        "background_color": "清除标志设为纯色时使用的背景颜色。",
        "pixel_width": "渲染目标宽度（像素）。",
        "pixel_height": "渲染目标高度（像素）。",
        "screen_to_world_point": "将屏幕空间坐标转换为世界坐标。",
        "world_to_screen_point": "将世界空间坐标转换为屏幕坐标。",
        "screen_point_to_ray": "从屏幕空间坐标向场景发出射线。",
        "on_draw_gizmos_selected": "选中时绘制摄像机视锥 Gizmo。",
    },
    "Light": {
        "_class_": "为场景提供照明的光源组件。",
        "light_type": "光源类型（0=方向光，1=点光源，2=聚光灯）。",
        "color": "光源颜色（RGB）。",
        "intensity": "光源强度。",
        "range": "点光源或聚光灯的照射范围。",
        "spot_angle": "聚光灯的锥角（度）。",
        "inner_spot_angle": "聚光灯内锥角（度）。",
        "shadow_type": "阴影类型（0=无，1=硬阴影，2=软阴影）。",
        "shadow_strength": "阴影强度。",
        "shadow_bias": "阴影偏移量。",
        "shadow_normal_bias": "阴影法线偏移量。",
        "shadow_near_plane": "阴影近裁剪面。",
        "cookie": "光源投影纹理。",
    },
    "MeshRenderer": {
        "_class_": "使用网格和材质渲染 3D 几何体的组件。",
        "materials": "此渲染器使用的材质列表。",
        "shared_materials": "共享材质列表。",
        "material": "此渲染器使用的主材质。",
        "shared_material": "共享的主材质。",
        "mesh": "要渲染的网格数据。",
        "enabled": "此渲染器是否已启用。",
        "cast_shadows": "是否投射阴影。",
        "receive_shadows": "是否接收阴影。",
    },
    "Material": {
        "_class_": "材质类。控制物体的视觉外观。",
        "native": "底层 C++ InfMaterial 对象。",
        "name": "材质的显示名称。",
        "guid": "材质的全局唯一标识符。",
        "render_queue": "渲染队列优先级，用于绘制排序。",
        "shader_name": "材质使用的着色器程序名称。",
        "vert_shader_name": "顶点着色器名称。",
        "frag_shader_name": "片段着色器名称。",
        "is_builtin": "是否为引擎内置材质。",
        "render_state_overrides": "应用于此材质的渲染状态覆盖位掩码。",
        "cull_mode": "面剔除模式（0=无，1=正面，2=背面）。",
        "depth_write_enable": "是否启用深度缓冲写入。",
        "depth_test_enable": "是否启用深度测试。",
        "depth_compare_op": "深度比较运算符。",
        "blend_enable": "是否启用 Alpha 混合。",
        "surface_type": "表面类型（'opaque' 或 'transparent'）。",
        "alpha_clip_enabled": "是否启用 Alpha 裁剪。",
        "alpha_clip_threshold": "Alpha 裁剪阈值。",
        "dispose": "释放底层原生材质资源。",
        "set_shader": "设置材质使用的着色器。",
        "set_float": "设置浮点数 uniform 属性。",
        "set_int": "设置整数 uniform 属性。",
        "set_color": "设置颜色 uniform 属性。",
        "set_vector2": "设置二维向量 uniform 属性。",
        "set_vector3": "设置三维向量 uniform 属性。",
        "set_vector4": "设置四维向量 uniform 属性。",
        "set_texture_guid": "通过 GUID 将纹理分配给采样器槽。",
        "clear_texture": "移除分配给采样器槽的纹理。",
        "get_float": "获取浮点数属性值。",
        "get_int": "获取整数属性值。",
        "get_color": "获取颜色属性（返回 RGBA 元组）。",
        "get_vector2": "获取二维向量属性。",
        "get_vector3": "获取三维向量属性。",
        "get_vector4": "获取四维向量属性。",
        "get_texture": "获取采样器槽中纹理的 GUID。",
        "has_property": "检查着色器属性是否存在。",
        "get_property": "按名称获取着色器属性值。",
        "get_all_properties": "获取所有着色器属性的字典。",
        "to_dict": "将材质序列化为字典。",
        "save": "将材质保存到文件。",
        "create_lit": "使用默认 PBR 着色器创建新材质。",
        "create_unlit": "使用无光照着色器创建新材质。",
        "from_native": "封装现有的 C++ InfMaterial 实例。",
        "load": "从文件路径加载材质。",
        "get": "按名称获取缓存的材质。",
    },
    "Shader": {
        "_class_": "着色器程序资源。",
        "name": "着色器名称。",
        "guid": "着色器的全局唯一标识符。",
        "is_valid": "着色器是否有效。",
        "find": "按名称查找着色器。",
        "load": "从文件加载着色器。",
    },
    "Texture": {
        "_class_": "纹理资源。",
        "name": "纹理名称。",
        "guid": "纹理的全局唯一标识符。",
        "width": "纹理宽度（像素）。",
        "height": "纹理高度（像素）。",
        "is_valid": "纹理是否有效。",
        "load": "从文件路径加载纹理。",
        "find": "按名称查找纹理。",
        "get": "按名称获取缓存的纹理。",
    },
    "MeshData": {
        "_class_": "网格数据，包含顶点、索引和属性。",
        "name": "网格名称。",
        "guid": "网格的全局唯一标识符。",
        "vertex_count": "顶点数量。",
        "index_count": "索引数量。",
        "is_valid": "网格数据是否有效。",
        "load": "从文件加载网格数据。",
        "find": "按名称查找网格数据。",
        "create_primitive": "创建基本几何体网格。",
    },
    "vector2": {
        "_class_": "二维向量，包含 x 和 y 分量。",
        "x": "X 分量。",
        "y": "Y 分量。",
        "magnitude": "向量的长度。",
        "sqr_magnitude": "向量长度的平方。",
        "normalized": "返回单位化的向量。",
        "zero": "Vector2(0, 0)。",
        "one": "Vector2(1, 1)。",
        "up": "Vector2(0, 1)。",
        "down": "Vector2(0, -1)。",
        "left": "Vector2(-1, 0)。",
        "right": "Vector2(1, 0)。",
        "normalize": "将此向量单位化。",
        "dot": "计算两个向量的点积。",
        "distance": "计算两点之间的距离。",
        "lerp": "在两个向量之间线性插值。",
        "angle": "计算两个向量之间的角度。",
    },
    "vector3": {
        "_class_": "三维向量，包含 x、y 和 z 分量。",
        "x": "X 分量。",
        "y": "Y 分量。",
        "z": "Z 分量。",
        "magnitude": "向量的长度。",
        "sqr_magnitude": "向量长度的平方。",
        "normalized": "返回单位化的向量。",
        "zero": "Vector3(0, 0, 0)。",
        "one": "Vector3(1, 1, 1)。",
        "up": "Vector3(0, 1, 0)。",
        "down": "Vector3(0, -1, 0)。",
        "left": "Vector3(-1, 0, 0)。",
        "right": "Vector3(1, 0, 0)。",
        "forward": "Vector3(0, 0, 1)。",
        "back": "Vector3(0, 0, -1)。",
        "normalize": "将此向量单位化。",
        "dot": "计算两个向量的点积。",
        "cross": "计算两个向量的叉积。",
        "distance": "计算两点之间的距离。",
        "lerp": "在两个向量之间线性插值。",
        "angle": "计算两个向量之间的角度。",
    },
    "vector4": {
        "_class_": "四维向量，包含 x、y、z 和 w 分量。",
        "x": "X 分量。",
        "y": "Y 分量。",
        "z": "Z 分量。",
        "w": "W 分量。",
        "magnitude": "向量的长度。",
        "sqr_magnitude": "向量长度的平方。",
        "normalized": "返回单位化的向量。",
        "zero": "Vector4(0, 0, 0, 0)。",
        "one": "Vector4(1, 1, 1, 1)。",
        "normalize": "将此向量单位化。",
        "dot": "计算两个向量的点积。",
        "distance": "计算两点之间的距离。",
        "lerp": "在两个向量之间线性插值。",
    },
    "Input": {
        "_class_": "用于读取键盘、鼠标和触摸输入的接口。",
        "mouse_position": "当前鼠标在屏幕坐标中的位置。",
        "game_mouse_position": "当前鼠标在游戏视口坐标中的位置。",
        "mouse_scroll_delta": "当前帧的鼠标滚轮增量。",
        "input_string": "当前帧中用户输入的字符。",
        "any_key": "当任意键或鼠标按钮被按住时返回 True。",
        "any_key_down": "当任意键或鼠标按钮首次按下时返回 True。",
        "touch_count": "当前活动的触摸数量。",
        "set_game_focused": "设置游戏视口是否获得输入焦点。",
        "set_game_viewport_origin": "设置游戏视口原点的屏幕坐标。",
        "is_game_focused": "游戏视口是否获得输入焦点。",
        "get_key": "当用户按住指定按键时返回 True。",
        "get_key_down": "当用户按下指定按键的那一帧返回 True。",
        "get_key_up": "当用户松开指定按键的那一帧返回 True。",
        "get_mouse_button": "当鼠标按钮被按住时返回 True。",
        "get_mouse_button_down": "当鼠标按钮按下的那一帧返回 True。",
        "get_mouse_button_up": "当鼠标按钮松开的那一帧返回 True。",
        "get_mouse_position": "获取当前鼠标在屏幕坐标中的位置。",
        "get_mouse_scroll_delta": "获取当前帧的鼠标滚轮增量。",
        "get_axis": "返回经过平滑处理的虚拟轴的值。",
        "get_axis_raw": "返回未经平滑处理的虚拟轴原始值。",
        "get_input_string": "获取当前帧中用户输入的字符。",
        "reset_input_axes": "将所有输入轴重置为零。",
    },
    "KeyCode": {
        "_class_": "按键代码枚举，用于标识键盘和鼠标按键。",
    },
    "Debug": {
        "_class_": "调试工具类。",
        "log": "输出日志消息到控制台。",
        "log_warning": "输出警告消息到控制台。",
        "log_error": "输出错误消息到控制台。",
        "draw_line": "在场景中绘制一条调试线段。",
        "draw_ray": "在场景中绘制一条调试射线。",
    },
    "Gizmos": {
        "_class_": "在场景视图中绘制调试可视化图形的工具类。",
        "color": "下一次绘制操作使用的颜色。",
        "draw_line": "绘制一条线段。",
        "draw_wire_sphere": "绘制线框球体。",
        "draw_sphere": "绘制实心球体。",
        "draw_wire_cube": "绘制线框立方体。",
        "draw_cube": "绘制实心立方体。",
        "draw_ray": "绘制一条射线。",
        "draw_frustum": "绘制视锥体。",
        "draw_icon": "在指定位置绘制图标。",
    },
    "Rigidbody": {
        "_class_": "刚体组件。让物体受物理引擎控制——牛顿看了都点头。",
        "mass": "刚体质量（千克）。",
        "drag": "线性阻力。",
        "angular_drag": "角阻力。",
        "use_gravity": "是否受重力影响。",
        "is_kinematic": "是否为运动学模式（不受力影响，但能推动别人）。",
        "constraints": "冻结哪些轴的位置或旋转。",
        "collision_detection_mode": "碰撞检测模式。",
        "interpolation": "插值模式。",
        "velocity": "线速度。",
        "angular_velocity": "角速度。",
        "position": "刚体位置。",
        "rotation": "刚体旋转。",
        "add_force": "施加力。",
        "add_torque": "施加扭矩。",
        "add_force_at_position": "在指定位置施加力。",
        "move_position": "移动刚体到目标位置。",
        "move_rotation": "旋转刚体到目标朝向。",
        "sleep": "强制刚体进入休眠。",
        "wake_up": "唤醒刚体。",
        "is_sleeping": "刚体是否正在休眠。",
    },
    "Collider": {
        "_class_": "碰撞体基类。所有碰撞体的老祖宗。",
        "is_trigger": "是否为触发器模式。",
        "center": "碰撞体在本地空间的中心偏移。",
        "enabled": "碰撞体是否启用。",
    },
    "BoxCollider": {
        "_class_": "盒形碰撞体。适合箱子、墙壁等方方正正的东西。",
        "size": "盒体的尺寸（本地空间）。",
        "center": "盒体中心偏移。",
    },
    "SphereCollider": {
        "_class_": "球形碰撞体。球是上帝最爱的形状。",
        "radius": "碰撞球半径。",
        "center": "球心偏移。",
    },
    "CapsuleCollider": {
        "_class_": "胶囊碰撞体。角色控制器的好搭档。",
        "radius": "胶囊半径。",
        "height": "胶囊高度。",
        "center": "胶囊中心偏移。",
    },
    "MeshCollider": {
        "_class_": "网格碰撞体。用真实网格做碰撞——精确但费性能。",
        "convex": "是否使用凸包近似。",
    },
    "Physics": {
        "_class_": "物理系统的静态工具类。",
        "raycast": "从原点沿方向发射射线检测碰撞。",
        "raycast_all": "射线检测所有碰撞体。",
        "overlap_sphere": "检测球形区域内的所有碰撞体。",
        "gravity": "全局重力加速度。",
    },
    "AudioSource": {
        "_class_": "音频源组件。在场景中播放声音的扬声器。",
        "clip": "当前音频剪辑。",
        "volume": "音量（0.0 到 1.0）。",
        "pitch": "音调。",
        "loop": "是否循环播放。",
        "play_on_awake": "是否在 Awake 时自动播放。",
        "spatial_blend": "空间混合（0=2D, 1=3D）。",
        "min_distance": "3D 声音的最小距离。",
        "max_distance": "3D 声音的最大距离。",
        "is_playing": "当前是否正在播放。",
        "time": "当前播放位置（秒）。",
        "play": "播放音频。",
        "pause": "暂停。",
        "unpause": "继续播放。",
        "stop": "停止。",
        "play_one_shot": "播放一次性音效（不影响主 clip）。",
    },
    "AudioListener": {
        "_class_": "音频监听器组件。场景中的耳朵——通常挂在主摄像机上。",
        "enabled": "监听器是否启用。",
    },
    "AudioClip": {
        "_class_": "音频剪辑资源。",
        "name": "剪辑名称。",
        "guid": "全局唯一标识符。",
        "duration": "时长（秒）。",
        "channels": "声道数。",
        "sample_rate": "采样率。",
        "load": "从文件加载音频剪辑。",
        "find": "按名称查找音频剪辑。",
    },
    "UICanvas": {
        "_class_": "UI 画布组件。所有 UI 元素的根容器——UI 的舞台。",
        "render_mode": "渲染模式。",
        "sort_order": "排序顺序。",
        "reference_resolution": "参考分辨率。",
    },
    "UIText": {
        "_class_": "UI 文本组件。在屏幕上显示文字。",
        "text": "显示的文本内容。",
        "font_size": "字体大小。",
        "color": "文本颜色。",
        "alignment_h": "水平对齐方式。",
        "alignment_v": "垂直对齐方式。",
    },
    "UIImage": {
        "_class_": "UI 图片组件。在屏幕上显示图片或色块。",
        "color": "图片颜色/叠色。",
        "texture_guid": "纹理的 GUID。",
        "raycast_target": "是否响应射线检测。",
    },
    "UIButton": {
        "_class_": "UI 按钮组件。用户点击的地方——程序员 Debug 的地方。",
        "on_click": "点击事件。",
        "interactable": "是否可交互。",
    },
    "UISelectable": {
        "_class_": "可选择的 UI 元素基类。UIButton 的老爸。",
        "interactable": "是否可交互。",
        "transition": "过渡类型。",
        "normal_color": "常态颜色。",
        "highlighted_color": "高亮颜色。",
        "pressed_color": "按下颜色。",
        "disabled_color": "禁用颜色。",
    },
    "InfUIComponent": {
        "_class_": "所有 UI 组件的 Python 基类。",
    },
    "PointerEventData": {
        "_class_": "指针事件数据。包含点击位置和来源信息。",
        "position": "当前指针屏幕坐标。",
        "button": "触发事件的鼠标按钮。",
        "click_count": "点击次数。",
    },
    "Time": {
        "_class_": "时间管理器。掌管每一帧的时间节奏——引擎的心跳。",
        "time": "自游戏启动以来的时间（秒）。",
        "delta_time": "上一帧耗时（秒）。写游戏逻辑离不开它。",
        "fixed_delta_time": "固定更新时间间隔（秒）。",
        "unscaled_time": "不受 time_scale 影响的时间。",
        "unscaled_delta_time": "不受 time_scale 影响的帧耗时。",
        "time_scale": "时间缩放。设为 0 暂停，设为 2 双倍速。",
        "frame_count": "自启动以来的帧数。",
        "realtime_since_startup": "自启动以来的真实时间（秒）。",
    },
    "Mathf": {
        "_class_": "数学工具类。常用数学函数大全。",
        "PI": "π（3.14159...）——数学界的摇滚巨星。",
        "Infinity": "正无穷。",
        "Epsilon": "极小正数。",
        "Deg2Rad": "度转弧度系数。",
        "Rad2Deg": "弧度转度系数。",
        "clamp": "将值限制在 min 和 max 之间。",
        "clamp01": "将值限制在 0 和 1 之间。",
        "lerp": "线性插值。",
        "inverse_lerp": "反向线性插值。",
        "move_towards": "向目标移动指定步长。",
        "smooth_step": "平滑插值（Hermite 曲线）。",
        "sign": "返回值的符号（-1 / 0 / 1）。",
        "abs": "绝对值。",
        "min": "返回较小值。",
        "max": "返回较大值。",
        "floor": "向下取整。",
        "ceil": "向上取整。",
        "round": "四舍五入。",
        "sqrt": "平方根。",
        "pow": "乘方。",
        "sin": "正弦。",
        "cos": "余弦。",
        "tan": "正切。",
        "asin": "反正弦。",
        "acos": "反余弦。",
        "atan": "反正切。",
        "atan2": "双参数反正切。",
    },
    "Coroutine": {
        "_class_": "协程句柄。代表一个正在运行的协程。",
        "is_done": "协程是否已完成。",
    },
    "WaitForSeconds": {
        "_class_": "等待指定秒数（受 Time.time_scale 影响）。",
        "duration": "等待时长（秒）。",
    },
    "WaitForSecondsRealtime": {
        "_class_": "等待指定真实秒数（不受 time_scale 影响）。暂停菜单的好朋友。",
        "duration": "等待时长（秒）。",
    },
    "WaitForEndOfFrame": {
        "_class_": "等待到当前帧渲染结束。截屏用得上。",
    },
    "WaitForFixedUpdate": {
        "_class_": "等待到下一次 FixedUpdate。物理相关计算适合在这里等。",
    },
    "WaitUntil": {
        "_class_": "等待直到条件为 True。",
        "predicate": "判断条件的可调用对象。",
    },
    "WaitWhile": {
        "_class_": "等待只要条件为 True 就继续等（条件变 False 时恢复）。",
        "predicate": "判断条件的可调用对象。",
    },
    "RenderGraph": {
        "_class_": "声明式渲染图。用 Pass 描述你想怎么画，引擎帮你调度。",
        "add_pass": "添加一个渲染 Pass。",
        "create_texture": "创建临时纹理。",
        "import_texture": "导入外部纹理。",
        "execute": "编译并执行渲染图。",
    },
    "RenderPassBuilder": {
        "_class_": "渲染 Pass 构建器。链式 API 定义输入输出。",
        "read": "声明此 Pass 读取某纹理。",
        "write": "声明此 Pass 写入某纹理。",
        "set_render_func": "设置此 Pass 的渲染回调。",
    },
    "TextureHandle": {
        "_class_": "渲染图中的临时纹理句柄。",
    },
    "RenderStack": {
        "_class_": "后处理效果栈。管理一系列后处理 Pass 的执行顺序。",
    },
    "RenderPass": {
        "_class_": "自定义渲染 Pass 的基类。",
        "setup": "在此配置 Pass 所需资源。",
        "execute": "执行渲染逻辑。",
    },
    "RenderPipeline": {
        "_class_": "可编程渲染管线基类。继承它来定制整个渲染流程。",
        "render": "每帧调用，执行渲染。",
    },
    "FullScreenEffect": {
        "_class_": "全屏后处理效果基类。自定义后处理从这里继承。",
    },
    "BloomEffect": {
        "_class_": "泛光效果。让亮处溢出光晕——梦幻感拉满。",
        "threshold": "亮度阈值。",
        "intensity": "泛光强度。",
        "scatter": "散射范围。",
    },
    "ToneMappingEffect": {
        "_class_": "色调映射效果。把 HDR 颜色压到屏幕可显示范围。",
        "mode": "映射模式（ACES / Reinhard / Neutral 等）。",
    },
    "VignetteEffect": {
        "_class_": "暗角效果。画面四周渐暗——电影感利器。",
        "intensity": "暗角强度。",
        "smoothness": "过渡平滑度。",
    },
    "ColorAdjustmentsEffect": {
        "_class_": "色彩调整效果。亮度、对比度、饱和度一把抓。",
        "exposure": "曝光。",
        "contrast": "对比度。",
        "saturation": "饱和度。",
    },
    "ChromaticAberrationEffect": {
        "_class_": "色差效果。模拟镜头边缘的 RGB 偏移。",
        "intensity": "色差强度。",
    },
    "FilmGrainEffect": {
        "_class_": "胶片噪点效果。复古胶片质感。",
        "intensity": "噪点强度。",
    },
    "SharpenEffect": {
        "_class_": "锐化效果。让画面更清晰。",
        "intensity": "锐化强度。",
    },
    "WhiteBalanceEffect": {
        "_class_": "白平衡效果。调节色温和色调。",
        "temperature": "色温。",
        "tint": "色调偏移。",
    },
    "quaternion": {
        "_class_": "四元数，表示三维旋转。比欧拉角靠谱，不会万向锁。",
        "x": "X 分量。",
        "y": "Y 分量。",
        "z": "Z 分量。",
        "w": "W 分量。",
        "identity": "单位四元数（无旋转）。",
        "euler": "从欧拉角创建四元数。",
        "angle_axis": "从轴角创建四元数。",
        "look_rotation": "创建朝向目标方向的旋转。",
        "slerp": "球面插值。",
        "inverse": "求逆。",
        "euler_angles": "转为欧拉角。",
    },
    "Format": {
        "_class_": "纹理格式枚举。",
    },
    "Space": {
        "_class_": "坐标空间枚举。World=世界空间，Self=本地空间。",
    },
    "ForceMode": {
        "_class_": "力的施加模式枚举。",
    },
    "PrimitiveType": {
        "_class_": "基本几何体类型枚举。",
    },
    "LightType": {
        "_class_": "光源类型枚举。",
    },
    "LightShadows": {
        "_class_": "光源阴影模式枚举。",
    },
    "CameraClearFlags": {
        "_class_": "摄像机清除模式枚举。",
    },
    "CameraProjection": {
        "_class_": "摄像机投影模式枚举。",
    },
    "LogLevel": {
        "_class_": "日志级别枚举。",
    },
    "RigidbodyConstraints": {
        "_class_": "刚体约束枚举。冻结指定轴的位置或旋转。",
    },
    "CollisionDetectionMode": {
        "_class_": "碰撞检测模式枚举。",
    },
}

def t(key: str, lang: str) -> str:
    """Translate a key into the given language."""
    return I18N.get(key, {}).get(lang, key)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    name: str
    type_hint: str = ""
    default: str = ""
    doc: str = ""

@dataclass
class MethodInfo:
    name: str
    params: List[ParamInfo] = field(default_factory=list)
    return_type: str = ""
    doc: str = ""
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    is_setter: bool = False
    is_operator: bool = False
    overloads: List["MethodInfo"] = field(default_factory=list)

@dataclass
class EnumValue:
    name: str
    value: str = ""
    doc: str = ""

@dataclass
class ClassInfo:
    name: str
    module: str                         # e.g. "InfEngine" or "InfEngine.core"
    doc: str = ""
    bases: List[str] = field(default_factory=list)
    properties: List[MethodInfo] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)
    static_methods: List[MethodInfo] = field(default_factory=list)
    constructors: List[MethodInfo] = field(default_factory=list)
    operators: List[MethodInfo] = field(default_factory=list)
    enum_values: List[EnumValue] = field(default_factory=list)
    is_enum: bool = False
    kind: str = "class"                 # "class", "enum", "function", "decorator"
    nested_enums: List["ClassInfo"] = field(default_factory=list)
    lifecycle_methods: List[MethodInfo] = field(default_factory=list)

@dataclass
class FunctionInfo:
    name: str
    module: str
    params: List[ParamInfo] = field(default_factory=list)
    return_type: str = ""
    doc: str = ""
    kind: str = "function"              # "function" or "decorator"

@dataclass
class ModuleInfo:
    name: str                           # e.g. "InfEngine.core"
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    doc: str = ""


# ---------------------------------------------------------------------------
# AST-based .pyi / .py introspection
# ---------------------------------------------------------------------------

OPERATOR_NAMES = {
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__iadd__", "__isub__", "__imul__",
    "__itruediv__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__",
    "__ge__", "__neg__", "__pos__", "__abs__", "__getitem__", "__setitem__",
    "__len__", "__contains__", "__repr__", "__str__", "__hash__",
    "__bool__", "__iter__",
}

LIFECYCLE_NAMES = {
    "awake", "start", "update", "fixed_update", "late_update",
    "on_destroy", "on_enable", "on_disable", "on_validate", "reset",
    "on_after_deserialize", "on_before_serialize",
    "on_draw_gizmos", "on_draw_gizmos_selected",
}


def _unparse_annotation(node) -> str:
    """Convert an AST annotation node back to source string."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _get_docstring(node) -> str:
    """Extract docstring from an AST class/function body."""
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        val = node.body[0].value
        if isinstance(val.value, str):
            return _clean_docstring(val.value)
    return ""


def _parse_func(node: ast.FunctionDef) -> MethodInfo:
    """Parse a function/method AST node into MethodInfo."""
    params = []
    args = node.args
    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        pi = ParamInfo(name=arg.arg)
        pi.type_hint = _unparse_annotation(arg.annotation)
        di = i - defaults_offset
        if di >= 0 and di < len(args.defaults):
            pi.default = ast.unparse(args.defaults[di])
        params.append(pi)

    # keyword-only args
    kw_defaults = args.kw_defaults
    for i, arg in enumerate(args.kwonlyargs):
        pi = ParamInfo(name=arg.arg)
        pi.type_hint = _unparse_annotation(arg.annotation)
        if kw_defaults[i] is not None:
            pi.default = ast.unparse(kw_defaults[i])
        params.append(pi)

    return_type = _unparse_annotation(node.returns)
    doc = _get_docstring(node)
    is_static = any(
        isinstance(d, ast.Name) and d.id == "staticmethod"
        for d in node.decorator_list
    )
    is_classmethod = any(
        isinstance(d, ast.Name) and d.id == "classmethod"
        for d in node.decorator_list
    )
    is_property = any(
        isinstance(d, ast.Name) and d.id == "property"
        for d in node.decorator_list
    )
    is_setter = any(
        isinstance(d, ast.Attribute) and d.attr == "setter"
        for d in node.decorator_list
    )
    is_operator = node.name in OPERATOR_NAMES

    return MethodInfo(
        name=node.name,
        params=params,
        return_type=return_type,
        doc=doc,
        is_static=is_static,
        is_classmethod=is_classmethod,
        is_property=is_property,
        is_setter=is_setter,
        is_operator=is_operator,
    )


def _parse_class(node: ast.ClassDef, module: str) -> ClassInfo:
    """Parse a class AST node into ClassInfo."""
    bases = [ast.unparse(b) for b in node.bases]
    doc = _get_docstring(node)
    is_enum = any(b in ("IntEnum", "Enum", "enum.IntEnum") for b in bases)

    ci = ClassInfo(
        name=node.name,
        module=module,
        doc=doc,
        bases=bases,
        is_enum=is_enum,
        kind="enum" if is_enum else "class",
    )

    # Track property names that have setters (to skip setter entries)
    setter_names = set()
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            for d in item.decorator_list:
                if isinstance(d, ast.Attribute) and d.attr == "setter":
                    setter_names.add(item.name)

    # Track overloaded methods
    overload_map: Dict[str, List[MethodInfo]] = {}
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            is_overload = any(
                (isinstance(d, ast.Name) and d.id == "overload") or
                (isinstance(d, ast.Attribute) and d.attr == "overload")
                for d in item.decorator_list
            )
            if is_overload:
                overload_map.setdefault(item.name, []).append(_parse_func(item))

    seen_methods = set()
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            mi = _parse_func(item)

            # Skip setter duplicates
            if mi.is_setter:
                continue

            # Skip private/internal methods
            if mi.name.startswith("_") and not mi.is_operator and mi.name != "__init__":
                continue

            # Skip overload-decorated versions (we'll aggregate them)
            is_overload = any(
                (isinstance(d, ast.Name) and d.id == "overload") or
                (isinstance(d, ast.Attribute) and d.attr == "overload")
                for d in item.decorator_list
            )
            if is_overload:
                continue

            # Attach overloads if any
            if mi.name in overload_map:
                mi.overloads = overload_map[mi.name]

            # Avoid duplicate entries
            if mi.name in seen_methods:
                continue
            seen_methods.add(mi.name)

            # Determine if property has setter
            has_setter = mi.name in setter_names

            if mi.name == "__init__":
                if mi.overloads:
                    ci.constructors.extend(mi.overloads)
                else:
                    ci.constructors.append(mi)
            elif mi.is_property:
                mi._has_setter = has_setter  # type: ignore
                ci.properties.append(mi)
            elif mi.is_operator:
                ci.operators.append(mi)
            elif mi.name in LIFECYCLE_NAMES:
                ci.lifecycle_methods.append(mi)
            elif mi.is_static or mi.is_classmethod:
                ci.static_methods.append(mi)
            else:
                ci.methods.append(mi)

        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            # Class-level annotated attribute (e.g. `x: float`)
            name = item.target.id
            if name.startswith("_"):
                continue
            type_hint = _unparse_annotation(item.annotation)
            doc = ""
            # Check if next node is a docstring expression
            idx = node.body.index(item)
            if idx + 1 < len(node.body):
                nxt = node.body[idx + 1]
                if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
                    doc = nxt.value.value.strip()

            if is_enum:
                ci.enum_values.append(EnumValue(name=name, value="", doc=doc))
            else:
                has_setter = name in setter_names
                ci.properties.append(MethodInfo(
                    name=name,
                    return_type=type_hint,
                    doc=doc,
                    is_property=True,
                ))
                ci.properties[-1]._has_setter = has_setter  # type: ignore

        elif isinstance(item, ast.Assign):
            # Enum values like `Cube: int`   or  `Cube = 0`
            for target in item.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    if is_enum:
                        ci.enum_values.append(EnumValue(
                            name=target.id,
                            value=ast.unparse(item.value) if item.value else "",
                        ))

        elif isinstance(item, ast.ClassDef):
            # Nested enum/class
            nested = _parse_class(item, module)
            ci.nested_enums.append(nested)

    return ci


def parse_stub_file(path: Path, module: str) -> ModuleInfo:
    """Parse a .pyi or .py file and extract all public classes and functions."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    mi = ModuleInfo(name=module)

    # Module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr):
        val = tree.body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            mi.doc = val.value.strip()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            ci = _parse_class(node, module)
            mi.classes.append(ci)

        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            fi = _parse_func(node)
            mi.functions.append(FunctionInfo(
                name=fi.name,
                module=module,
                params=fi.params,
                return_type=fi.return_type,
                doc=fi.doc,
            ))

    return mi


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _sig(mi: MethodInfo, class_name: str = "") -> str:
    """Build a human-readable method signature."""
    parts = []
    for p in mi.params:
        s = p.name
        if p.type_hint:
            s += f": {p.type_hint}"
        if p.default:
            s += f" = {p.default}"
        parts.append(s)
    args = ", ".join(parts)
    ret = f" → {mi.return_type}" if mi.return_type else ""
    prefix = f"{class_name}." if class_name else ""
    if mi.is_static:
        prefix = f"static {prefix}"
    return f"`{prefix}{mi.name}({args}){ret}`"


def _clean_docstring(doc: str) -> str:
    """Clean a raw docstring: strip, dedent, remove code examples for display."""
    if not doc:
        return ""
    # Standard Python docstring cleaning: inspect.cleandoc handles
    # the first-line-no-indent + rest-indented pattern correctly
    return inspect.cleandoc(doc)


def _short_doc(doc: str) -> str:
    """Extract the first sentence of a docstring."""
    if not doc:
        return ""
    # Filter out "Ellipsis" which comes from `...` in stubs
    if doc.strip() == "Ellipsis" or doc.strip() == "...":
        return ""
    first = doc.split("\n")[0].strip()
    # Truncate at first period if reasonable
    if ". " in first:
        first = first[:first.index(". ") + 1]
    return first


def _short_doc_zh(class_name: str, member_name: str, fallback: str) -> str:
    """Get Chinese short description, falling back to English."""
    cls_map = ZH_SHORT.get(class_name, {})
    zh = cls_map.get(member_name, "")
    if zh:
        return zh
    return fallback


def _user_block(section_id: str, existing_blocks: Dict[str, str]) -> str:
    """Generate a user content block, preserving existing content if available."""
    content = existing_blocks.get(section_id, "")
    return f"{USER_START} {section_id}\n{content}\n{USER_END}"


def _extract_user_blocks(text: str) -> Dict[str, str]:
    """Extract existing user-written content blocks from an MD file."""
    blocks = {}
    pattern = re.compile(
        rf"{re.escape(USER_START)}\s*(\S+)\n(.*?)\n{re.escape(USER_END)}",
        re.DOTALL
    )
    for m in pattern.finditer(text):
        blocks[m.group(1)] = m.group(2)
    return blocks


def generate_class_page(ci: ClassInfo, lang: str, existing: str = "") -> str:
    """Generate a single class/enum API reference page."""
    blocks = _extract_user_blocks(existing)
    lines: List[str] = []

    # Title
    lines.append(f"# {ci.name}\n")

    # Class header
    kind_key = "enum_in" if ci.is_enum else "class_in"
    lines.append(f'<div class="class-info">')
    lines.append(f'{t(kind_key, lang)} <b>{ci.module}</b>')
    lines.append(f'</div>\n')

    # Inheritance
    _base_link_map = {"BuiltinComponent": "Component"}
    if ci.bases and not ci.is_enum:
        non_trivial = [b for b in ci.bases if b not in ("object", "IntEnum", "Enum")]
        if non_trivial:
            parts = []
            for b in non_trivial:
                target = _base_link_map.get(b, b)
                parts.append(f'[{b}]({target}.md)')
            lines.append(f"**{t('inherits', lang)}:** {', '.join(parts)}\n")

    # Description
    lines.append(f"## {t('description', lang)}\n")
    if ci.doc:
        if lang == LANG_ZH:
            zh_class_doc = ZH_SHORT.get(ci.name, {}).get("_class_", "")
            lines.append(f"{zh_class_doc or ci.doc}\n")
        else:
            lines.append(f"{ci.doc}\n")
    lines.append(_user_block("description", blocks))
    lines.append("")

    # Enum values
    if ci.is_enum and ci.enum_values:
        lines.append(f"## {t('values', lang)}\n")
        lines.append(f"| {t('name', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|")
        for ev in ci.enum_values:
            doc = _short_doc(ev.doc) if ev.doc else ""
            lines.append(f"| {ev.name} | {doc} |")
        lines.append("")
        lines.append(_user_block("enum_values", blocks))
        lines.append("")

    # Constructors
    if ci.constructors:
        lines.append(f"## {t('constructors', lang)}\n")
        lines.append(f"| {t('signature', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|")
        for m in ci.constructors:
            sig = _sig(m, ci.name)
            if lang == LANG_ZH:
                doc = _short_doc_zh(ci.name, "__init__", _short_doc(m.doc))
            else:
                doc = _short_doc(m.doc)
            lines.append(f"| {sig} | {doc} |")
        lines.append("")
        lines.append(_user_block("constructors", blocks))
        lines.append("")

    # Properties
    if ci.properties:
        lines.append(f"## {t('properties', lang)}\n")
        lines.append(f"| {t('name', lang)} | {t('type', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|------|")
        for p in ci.properties:
            rw = ""
            if hasattr(p, "_has_setter") and not p._has_setter:
                rw = " *(read-only)*" if lang == "en" else " *(只读)*"
            type_str = p.return_type if p.return_type else ""
            if lang == LANG_ZH:
                doc = _short_doc_zh(ci.name, p.name, _short_doc(p.doc))
            else:
                doc = _short_doc(p.doc)
            lines.append(f"| {p.name} | `{type_str}` | {doc}{rw} |")
        lines.append("")
        lines.append(_user_block("properties", blocks))
        lines.append("")

    # Public methods
    if ci.methods:
        lines.append(f"## {t('public_methods', lang)}\n")
        lines.append(f"| {t('method', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|")
        for m in ci.methods:
            sig = _sig(m)
            if lang == LANG_ZH:
                doc = _short_doc_zh(ci.name, m.name, _short_doc(m.doc))
            else:
                doc = _short_doc(m.doc)
            lines.append(f"| {sig} | {doc} |")
        lines.append("")
        lines.append(_user_block("public_methods", blocks))
        lines.append("")

    # Static methods
    if ci.static_methods:
        lines.append(f"## {t('static_methods', lang)}\n")
        lines.append(f"| {t('method', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|")
        for m in ci.static_methods:
            sig = _sig(m, ci.name)
            if lang == LANG_ZH:
                doc = _short_doc_zh(ci.name, m.name, _short_doc(m.doc))
            else:
                doc = _short_doc(m.doc)
            lines.append(f"| {sig} | {doc} |")
        lines.append("")
        lines.append(_user_block("static_methods", blocks))
        lines.append("")

    # Lifecycle methods (for InfComponent and subclasses)
    if ci.lifecycle_methods:
        lines.append(f"## {t('lifecycle_methods', lang)}\n")
        lines.append(f"| {t('method', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|")
        for m in ci.lifecycle_methods:
            sig = _sig(m)
            if lang == LANG_ZH:
                doc = _short_doc_zh(ci.name, m.name, _short_doc(m.doc))
            else:
                doc = _short_doc(m.doc)
            lines.append(f"| {sig} | {doc} |")
        lines.append("")
        lines.append(_user_block("lifecycle_methods", blocks))
        lines.append("")

    # Operators
    if ci.operators:
        lines.append(f"## {t('operators', lang)}\n")
        lines.append(f"| {t('method', lang)} | {t('returns', lang)} |")
        lines.append("|------|------|")
        for m in ci.operators:
            sig = _sig(m)
            ret = m.return_type if m.return_type else ""
            lines.append(f"| {sig} | `{ret}` |")
        lines.append("")
        lines.append(_user_block("operators", blocks))
        lines.append("")

    # Nested enums
    for ne in ci.nested_enums:
        lines.append(f"### {ne.name}\n")
        if ne.doc:
            lines.append(f"{ne.doc}\n")
        if ne.enum_values:
            lines.append(f"| {t('name', lang)} | {t('desc', lang)} |")
            lines.append("|------|------|")
            for ev in ne.enum_values:
                doc = _short_doc(ev.doc) if ev.doc else ""
                lines.append(f"| {ev.name} | {doc} |")
            lines.append("")

    # Example
    lines.append(f"## {t('example', lang)}\n")
    example_content = blocks.get("example", "").strip()
    if not example_content:
        lines.append("```python")
        lines.append(f"# TODO: Add example for {ci.name}")
        lines.append("```\n")
    lines.append(_user_block("example", blocks))
    lines.append("")

    # See Also
    lines.append(f"## {t('see_also', lang)}\n")
    lines.append(_user_block("see_also", blocks))
    lines.append("")

    return "\n".join(lines)


def generate_function_page(fi: FunctionInfo, lang: str, existing: str = "") -> str:
    """Generate a page for a standalone function or decorator."""
    blocks = _extract_user_blocks(existing)
    lines: List[str] = []

    lines.append(f"# {fi.name}\n")

    kind_key = "decorator_in" if fi.kind == "decorator" else "function_in"
    lines.append(f'<div class="class-info">')
    lines.append(f'{t(kind_key, lang)} <b>{fi.module}</b>')
    lines.append(f'</div>\n')

    # Signature
    parts = []
    for p in fi.params:
        s = p.name
        if p.type_hint:
            s += f": {p.type_hint}"
        if p.default:
            s += f" = {p.default}"
        parts.append(s)
    ret = f" → {fi.return_type}" if fi.return_type else ""
    lines.append(f"```python\n{fi.name}({', '.join(parts)}){ret}\n```\n")

    # Description
    lines.append(f"## {t('description', lang)}\n")
    if fi.doc:
        lines.append(f"{fi.doc}\n")
    lines.append(_user_block("description", blocks))
    lines.append("")

    # Parameters
    if fi.params:
        lines.append(f"## {t('parameters', lang)}\n")
        lines.append(f"| {t('name', lang)} | {t('type', lang)} | {t('desc', lang)} |")
        lines.append("|------|------|------|")
        for p in fi.params:
            default = f" (default: `{p.default}`)" if p.default else ""
            lines.append(f"| {p.name} | `{p.type_hint}` | {p.doc}{default} |")
        lines.append("")

    # Example
    lines.append(f"## {t('example', lang)}\n")
    example_content = blocks.get("example", "").strip()
    if not example_content:
        lines.append("```python")
        lines.append(f"# TODO: Add example for {fi.name}")
        lines.append("```\n")
    lines.append(_user_block("example", blocks))
    lines.append("")

    return "\n".join(lines)


def generate_index_page(modules: Dict[str, ModuleInfo], lang: str, existing: str = "") -> str:
    """Generate the API index page."""
    blocks = _extract_user_blocks(existing)
    lines: List[str] = []

    lines.append(f"# {t('api_ref_title', lang)}\n")
    lines.append(f'<div class="class-info">')
    lines.append(f'{t("version", lang)}')
    lang_other = "zh" if lang == "en" else "en"
    lang_labels = {"en": "English", "zh": "中文"}
    lines.append(f' &nbsp;|&nbsp; <a href="../../{lang_other}/api/index.html">{lang_labels[lang_other]}</a>')
    lines.append(f'</div>\n')

    lines.append(f"## {t('description', lang)}\n")
    lines.append(f"{t('api_ref_welcome', lang)}\n")
    lines.append(_user_block("description", blocks))
    lines.append("")

    lines.append(f"## {t('packages', lang)}\n")
    lines.append(f"| {t('package', lang)} | {t('desc', lang)} |")
    lines.append("|------|------|")

    for mod_name, mod in sorted(modules.items()):
        class_names = ", ".join(c.name for c in mod.classes[:6])
        if len(mod.classes) > 6:
            class_names += ", ..."
        func_names = ", ".join(f.name for f in mod.functions[:3])
        all_names = ", ".join(filter(None, [class_names, func_names]))
        lines.append(f"| {mod_name} | {all_names} |")

    lines.append("")
    lines.append(_user_block("index", blocks))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module discovery — auto-discovery + public API whitelist
# ---------------------------------------------------------------------------

# Directories to exclude when walking the package tree
AUTO_DISCOVER_EXCLUDE_DIRS = {
    "__pycache__",
    "engine",       # Editor UI — not part of the scripting API
    "examples",     # Example scripts
    "resources",    # Internal resource helpers
}

# ── Public API whitelist ──────────────────────────────────────────────────
# Only classes / enums / functions listed here produce documentation pages
# in the default mode.  Use ``--all`` to generate pages for everything.
#
# When you add a new user-facing type, add its name here so the docs are
# regenerated on the next build.
# ──────────────────────────────────────────────────────────────────────────

PUBLIC_API_CLASSES = {
    # ── Core ──
    "GameObject",
    "Transform",
    "Component",
    "InfComponent",
    "Scene",
    "SceneManager",

    # ── Rendering Components ──
    "Camera",
    "Light",
    "MeshRenderer",

    # ── Physics ──
    "Rigidbody",
    "Collider",
    "BoxCollider",
    "SphereCollider",
    "CapsuleCollider",
    "MeshCollider",
    "Physics",

    # ── Audio ──
    "AudioSource",
    "AudioListener",
    "AudioClip",

    # ── UI ──
    "UICanvas",
    "UIText",
    "UIImage",
    "UIButton",
    "UISelectable",
    "InfUIComponent",
    "PointerEventData",

    # ── Resources ──
    "Material",
    "Shader",
    "Texture",
    "MeshData",

    # ── Math ──
    "vector2",
    "vector3",
    "vector4",
    "quaternion",
    "Mathf",

    # ── Time & Coroutines ──
    "Time",
    "Coroutine",
    "WaitForSeconds",
    "WaitForSecondsRealtime",
    "WaitForEndOfFrame",
    "WaitForFixedUpdate",
    "WaitUntil",
    "WaitWhile",

    # ── Input ──
    "Input",
    "KeyCode",

    # ── Debug & Gizmos ──
    "Debug",
    "Gizmos",

    # ── Render Pipeline ──
    "RenderGraph",
    "RenderPassBuilder",
    "TextureHandle",
    "RenderStack",
    "RenderPass",
    "RenderPipeline",
    "FullScreenEffect",

    # ── Post-Processing Effects ──
    "BloomEffect",
    "ToneMappingEffect",
    "VignetteEffect",
    "ColorAdjustmentsEffect",
    "ChromaticAberrationEffect",
    "FilmGrainEffect",
    "SharpenEffect",
    "WhiteBalanceEffect",

    # ── Enums ──
    "RigidbodyConstraints",
    "CollisionDetectionMode",
    "ForceMode",
    "Space",
    "PrimitiveType",
    "LightType",
    "LightShadows",
    "CameraClearFlags",
    "CameraProjection",
    "LogLevel",
    "Format",
}

PUBLIC_API_FUNCTIONS = {
    # ── Decorators / Attributes ──
    "serialized_field",
    "require_component",
    "add_component_menu",
    "execute_in_edit_mode",
    "disallow_multiple",
    # help_url, hide_field, icon → too trivial for standalone pages
}


def _path_to_module(rel_path: Path) -> str:
    """Map a file path (relative to STUB_ROOT) to its logical module name.

    Examples
    --------
    core/material.py      → InfEngine.core
    input/__init__.py     → InfEngine.input
    lib/_InfEngine.pyi    → InfEngine
    debug.py              → InfEngine.debug
    gizmos/gizmos.py      → InfEngine.gizmos
    math/vector.py        → InfEngine.math
    __init__.py           → InfEngine
    """
    parts = list(rel_path.parts)
    stem = parts[-1]
    if stem.endswith(".pyi"):
        stem = stem[:-4]
    elif stem.endswith(".py"):
        stem = stem[:-3]

    # Special case: native bindings stub
    if len(parts) >= 2 and parts[-2] == "lib" and stem == "_InfEngine":
        return "InfEngine"

    # __init__ → module is the enclosing package
    if stem == "__init__":
        pkg_parts = parts[:-1]
        if not pkg_parts:
            return "InfEngine"
        return "InfEngine." + ".".join(pkg_parts)

    # Regular file → module is the enclosing package
    pkg_parts = parts[:-1]
    if not pkg_parts:
        return f"InfEngine.{stem}"
    return "InfEngine." + ".".join(pkg_parts)


def auto_discover_sources() -> List[Tuple[str, Path]]:
    """Walk *python/InfEngine/* and collect every parseable .pyi / .py file.

    When both ``foo.pyi`` and ``foo.py`` exist in the same directory the
    ``.pyi`` stub is preferred (it has cleaner type information).

    Returns a list of ``(module_name, absolute_path)`` tuples.
    """
    sources: List[Tuple[str, Path]] = []

    for dirpath_str, dirnames, filenames in os.walk(STUB_ROOT):
        dirpath = Path(dirpath_str)
        # Prune excluded subtrees
        dirnames[:] = sorted(d for d in dirnames if d not in AUTO_DISCOVER_EXCLUDE_DIRS)

        # Prefer .pyi over .py for the same stem
        file_map: Dict[str, Path] = {}
        for fname in sorted(filenames):
            if fname.endswith(".pyi"):
                stem = fname[:-4]
                file_map[stem] = dirpath / fname          # .pyi always wins
            elif fname.endswith(".py"):
                stem = fname[:-3]
                if stem not in file_map:                   # only when no .pyi
                    file_map[stem] = dirpath / fname

        for stem, fpath in sorted(file_map.items()):
            # Skip private files except the native bindings and __init__
            if stem.startswith("_") and stem != "_InfEngine" and stem != "__init__":
                continue

            rel = fpath.relative_to(STUB_ROOT)
            mod_name = _path_to_module(rel)
            sources.append((mod_name, fpath))

    return sources


def discover_modules(*, include_all: bool = False) -> Dict[str, ModuleInfo]:
    """Discover and parse all modules.

    Parameters
    ----------
    include_all : bool
        When *True*, every discovered class/function gets a page (no
        whitelist filtering).  Activated by the ``--all`` CLI flag.
    """
    sources = auto_discover_sources()
    modules: Dict[str, ModuleInfo] = {}

    for mod_name, path in sources:
        print(f"  Parsing {path.relative_to(PROJECT_ROOT)} → {mod_name}")
        parsed = parse_stub_file(path, mod_name)

        # Merge into existing module bucket
        if mod_name not in modules:
            modules[mod_name] = ModuleInfo(name=mod_name, doc=parsed.doc)

        mod = modules[mod_name]

        for ci in parsed.classes:
            if not include_all and ci.name not in PUBLIC_API_CLASSES:
                continue
            existing_names = {c.name for c in mod.classes}
            if ci.name not in existing_names:
                mod.classes.append(ci)

        for fi in parsed.functions:
            if not include_all and fi.name not in PUBLIC_API_FUNCTIONS:
                continue
            existing_names = {f.name for f in mod.functions}
            if fi.name not in existing_names:
                mod.functions.append(fi)

    # Drop modules that ended up empty after filtering
    modules = {k: v for k, v in modules.items() if v.classes or v.functions}

    return modules


# ---------------------------------------------------------------------------
# File I/O — write pages, preserve user content
# ---------------------------------------------------------------------------

def _read_existing(path: Path) -> str:
    """Read existing file content (empty string if not exists)."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _write_if_changed(path: Path, content: str):
    """Write content to file only if it changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _read_existing(path)
    if old.strip() == content.strip():
        return False
    path.write_text(content, encoding="utf-8")
    return True


def generate_all(*, include_all: bool = False):
    """Main entry point: discover modules, generate all pages.

    Parameters
    ----------
    include_all : bool
        When *True* every discovered class/function gets a page regardless
        of the PUBLIC_API whitelist.  Pass ``--all`` on the CLI to activate.
    """
    print("=" * 60)
    print("InfEngine API Docs Generator")
    if include_all:
        print("  (--all mode: generating pages for ALL discovered types)")
    print("=" * 60)

    print("\nDiscovering modules...")
    modules = discover_modules(include_all=include_all)

    total_classes = sum(len(m.classes) for m in modules.values())
    total_funcs = sum(len(m.functions) for m in modules.values())
    print(f"\nFound {len(modules)} modules, {total_classes} classes, {total_funcs} functions")

    # Collect all pages for nav generation
    nav_entries: Dict[str, List[Tuple[str, str]]] = {}  # module → [(display_name, filename)]

    for lang in (LANG_EN, LANG_ZH):
        api_dir = EN_API if lang == LANG_EN else ZH_API
        print(f"\nGenerating {lang.upper()} pages in {api_dir.relative_to(PROJECT_ROOT)}...")

        written = 0

        for mod_name, mod in sorted(modules.items()):
            if mod_name not in nav_entries:
                nav_entries[mod_name] = []

            # Generate class pages
            for ci in mod.classes:
                filename = f"{ci.name}.md"
                filepath = api_dir / filename
                existing = _read_existing(filepath)
                content = generate_class_page(ci, lang, existing)
                if _write_if_changed(filepath, content):
                    written += 1

                # Track for nav (only once)
                if lang == LANG_EN:
                    entry = (ci.name, filename)
                    if entry not in nav_entries[mod_name]:
                        nav_entries[mod_name].append(entry)

            # Generate function pages
            for fi in mod.functions:
                filename = f"{fi.name}.md"
                filepath = api_dir / filename
                existing = _read_existing(filepath)
                content = generate_function_page(fi, lang, existing)
                if _write_if_changed(filepath, content):
                    written += 1

                if lang == LANG_EN:
                    entry = (fi.name, filename)
                    if entry not in nav_entries[mod_name]:
                        nav_entries[mod_name].append(entry)

        # Index page
        idx_path = api_dir / "index.md"
        existing = _read_existing(idx_path)
        content = generate_index_page(modules, lang, existing)
        if _write_if_changed(idx_path, content):
            written += 1

        print(f"  {written} files written/updated")

        # Clean up stale .md files that are no longer generated
        generated_filenames = {"index.md"}
        for mod_name, entries in nav_entries.items():
            for _, filename in entries:
                generated_filenames.add(filename)

        removed = 0
        for existing_file in api_dir.glob("*.md"):
            if existing_file.name not in generated_filenames:
                existing_file.unlink()
                removed += 1
        if removed:
            print(f"  {removed} stale files removed")

    # Generate mkdocs nav fragment
    _generate_nav_fragment(nav_entries)

    # Generate mkdocs.yml
    _generate_mkdocs_yml(nav_entries)

    print("\nDone! ✓")


def _generate_nav_fragment(nav_entries: Dict[str, List[Tuple[str, str]]]):
    """Generate a YAML nav fragment file for reference."""
    lines = ["# Auto-generated API nav — copy into mkdocs.yml if needed", ""]

    for lang in (LANG_EN, LANG_ZH):
        prefix = f"{lang}/api"
        section_title = "API Reference" if lang == "en" else "API 参考手册"
        lines.append(f"# {section_title}")

        for mod_name, entries in sorted(nav_entries.items()):
            lines.append(f"#   {mod_name}:")
            for display, filename in sorted(entries):
                lines.append(f"#     - {display}: {prefix}/{filename}")
        lines.append("")

    path = WIKI_ROOT / "mkdocs_api_nav.yml"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nNav fragment written to {path.relative_to(PROJECT_ROOT)}")


def _generate_mkdocs_yml(nav_entries: Dict[str, List[Tuple[str, str]]]):
    """Regenerate the full mkdocs.yml with auto-generated API nav."""
    lines = []
    lines.append("site_name: InfEngine Scripting API")
    lines.append("site_description: InfEngine Game Engine Scripting API Reference")
    lines.append("site_author: InfEngine Team")
    lines.append("")
    lines.append("use_directory_urls: false")
    lines.append("")
    lines.append("theme:")
    lines.append("  name: material")
    lines.append("  custom_dir: theme")
    lines.append("  language: en")
    lines.append("")
    lines.append("nav:")
    lines.append("  - Home: index.md")

    for lang, section_title in [("en", "API Reference"), ("zh", "API 参考手册")]:
        prefix = f"{lang}/api"
        lines.append(f"  - {section_title}:")
        lines.append(f"    - Overview: {prefix}/index.md")

        for mod_name, entries in sorted(nav_entries.items()):
            lines.append(f"    - {mod_name}:")
            for display, filename in sorted(entries):
                lines.append(f"      - {display}: {prefix}/{filename}")

    lines.append("")
    lines.append("markdown_extensions:")
    lines.append("  - toc:")
    lines.append("      permalink: true")
    lines.append("  - pymdownx.highlight:")
    lines.append("      anchor_linenums: true")
    lines.append("      use_pygments: true")
    lines.append("      pygments_lang_class: true")
    lines.append("  - pymdownx.superfences")
    lines.append("  - pymdownx.inlinehilite")
    lines.append("")
    lines.append("plugins:")
    lines.append("  - search")
    lines.append("")

    path = WIKI_ROOT / "mkdocs.yml"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"mkdocs.yml regenerated at {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    include_all = "--all" in sys.argv
    generate_all(include_all=include_all)
