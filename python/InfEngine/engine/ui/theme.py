"""
InfEngine 编辑器主题 / InfEngine Editor Theme
==============================================

这是 InfEngine 编辑器的 **唯一主题配置文件**。
修改此文件中的颜色和尺寸值即可改变整个编辑器的外观。

This is the **single theme configuration file** for the InfEngine Editor.
Change colors and sizes here to restyle the entire editor.

结构 / Structure:
    1. 颜色空间工具 — sRGB → Linear 转换辅助函数
    2. ImGui 枚举镜像 — ImGuiCol, ImGuiStyleVar, etc.
    3. Theme 类 — 所有颜色 / 尺寸 / 图标 / 布局常量
    4. 样式推入辅助方法 — push/pop 便捷方法

所有颜色为 **线性空间 RGBA 元组** (float, 0-1)。
All colors are **linear-space RGBA tuples** (float, 0-1).

用法 / Usage::

    from InfEngine.engine.ui.theme import Theme, ImGuiCol

    Theme.push_ghost_button_style(ctx)
    ctx.button("Click me", on_click)
    ctx.pop_style_color(3)
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple

# 颜色类型别名 / Color type alias (R, G, B, A) all float [0..1]
RGBA = Tuple[float, float, float, float]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. 颜色空间工具 / Color Space Utilities                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def srgb_to_linear(s: float) -> float:
    """将单个 sRGB [0,1] 分量转换为线性空间。
    Convert a single sRGB [0,1] component to linear space."""
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4


def srgb3(r: float, g: float, b: float, a: float = 1.0) -> RGBA:
    """将 sRGB 0-1 分量转换为线性 RGBA 元组。
    Convert sRGB 0-1 components to linear RGBA tuple."""
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)


def hex_to_linear(hex_r: int, hex_g: int, hex_b: int, a: float = 1.0) -> RGBA:
    """将 0-255 sRGB 分量转换为线性 RGBA 元组。
    Convert 0-255 sRGB hex components to linear RGBA tuple."""
    return srgb3(hex_r / 255.0, hex_g / 255.0, hex_b / 255.0, a)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. ImGui 枚举镜像 / ImGui Enum Mirrors                                ║
# ║     必须与 imgui.h 中的枚举顺序完全一致                                   ║
# ║     Must match imgui.h enum order exactly                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ImGuiCol:
    """ImGui 颜色索引 — 对应 imgui.h ImGuiCol_ 枚举。"""
    Text                       = 0
    TextDisabled               = 1
    WindowBg                   = 2
    ChildBg                    = 3
    PopupBg                    = 4
    Border                     = 5
    BorderShadow               = 6
    FrameBg                    = 7
    FrameBgHovered             = 8
    FrameBgActive              = 9
    TitleBg                    = 10
    TitleBgActive              = 11
    TitleBgCollapsed           = 12
    MenuBarBg                  = 13
    ScrollbarBg                = 14
    ScrollbarGrab              = 15
    ScrollbarGrabHovered       = 16
    ScrollbarGrabActive        = 17
    CheckMark                  = 18
    SliderGrab                 = 19
    SliderGrabActive           = 20
    Button                     = 21
    ButtonHovered              = 22
    ButtonActive               = 23
    Header                     = 24
    HeaderHovered              = 25
    HeaderActive               = 26
    Separator                  = 27
    SeparatorHovered           = 28
    SeparatorActive            = 29
    ResizeGrip                 = 30
    ResizeGripHovered          = 31
    ResizeGripActive           = 32
    InputTextCursor            = 33
    TabHovered                 = 34
    Tab                        = 35
    TabSelected                = 36
    TabSelectedOverline        = 37
    TabDimmed                  = 38
    TabDimmedSelected          = 39
    TabDimmedSelectedOverline  = 40
    DockingPreview             = 41
    DockingEmptyBg             = 42
    PlotLines                  = 43
    PlotLinesHovered           = 44
    PlotHistogram              = 45
    PlotHistogramHovered       = 46
    TableHeaderBg              = 47
    TableBorderStrong          = 48
    TableBorderLight           = 49
    TableRowBg                 = 50
    TableRowBgAlt              = 51
    TextLink                   = 52
    TextSelectedBg             = 53
    TreeLines                  = 54
    DragDropTarget             = 55
    DragDropTargetBg           = 56
    UnsavedMarker              = 57
    NavCursor                  = 58
    NavWindowingHighlight      = 59
    NavWindowingDimBg          = 60
    ModalWindowDimBg           = 61


class ImGuiWindowFlags:
    """ImGui 窗口标志 — 对应 imgui.h ImGuiWindowFlags_ 枚举。"""
    NoTitleBar                  = 1 << 0
    NoResize                    = 1 << 1
    NoMove                      = 1 << 2
    NoScrollbar                 = 1 << 3
    NoScrollWithMouse           = 1 << 4
    NoCollapse                  = 1 << 5
    AlwaysAutoResize            = 1 << 6
    NoBackground                = 1 << 7
    NoSavedSettings             = 1 << 8
    NoMouseInputs               = 1 << 9
    NoFocusOnAppearing          = 1 << 12
    NoBringToFrontOnFocus       = 1 << 13
    NoNavInputs                 = 1 << 16
    NoNavFocus                  = 1 << 17
    UnsavedDocument             = 1 << 18
    NoDocking                   = 1 << 19
    NoNav                       = (1 << 16) | (1 << 17)
    NoDecoration                = NoTitleBar | NoResize | NoScrollbar | NoCollapse
    NoInputs                    = NoMouseInputs | NoNavInputs | NoNavFocus


class ImGuiTreeNodeFlags:
    """ImGui 树节点标志 — 对应 imgui.h ImGuiTreeNodeFlags_ 枚举。"""
    Selected                    = 1 << 0
    Framed                      = 1 << 1
    AllowOverlap                = 1 << 2
    NoTreePushOnOpen            = 1 << 3
    NoAutoOpenOnLog             = 1 << 4
    DefaultOpen                 = 1 << 5
    OpenOnDoubleClick           = 1 << 6
    OpenOnArrow                 = 1 << 7
    Leaf                        = 1 << 8
    Bullet                      = 1 << 9
    FramePadding                = 1 << 10
    SpanAvailWidth              = 1 << 11
    SpanFullWidth               = 1 << 12
    SpanAllColumns              = 1 << 13
    CollapsingHeader            = Framed | NoTreePushOnOpen | NoAutoOpenOnLog


class ImGuiMouseCursor:
    """ImGui 鼠标光标枚举。"""
    Arrow      = 0
    TextInput  = 1
    ResizeAll  = 2
    ResizeNS   = 3
    ResizeEW   = 4
    ResizeNESW = 5
    ResizeNWSE = 6
    Hand       = 7


class ImGuiStyleVar:
    """ImGui 样式变量索引 — 对应 imgui.h ImGuiStyleVar_ 枚举。"""
    Alpha                       = 0
    DisabledAlpha               = 1
    WindowPadding               = 2
    WindowRounding              = 3
    WindowBorderSize            = 4
    WindowMinSize               = 5
    WindowTitleAlign            = 6
    ChildRounding               = 7
    ChildBorderSize             = 8
    PopupRounding               = 9
    PopupBorderSize             = 10
    FramePadding                = 11
    FrameRounding               = 12
    FrameBorderSize             = 13
    ItemSpacing                 = 14
    ItemInnerSpacing            = 15
    IndentSpacing               = 16
    CellPadding                 = 17
    ScrollbarSize               = 18
    ScrollbarRounding           = 19
    ScrollbarPadding            = 20
    GrabMinSize                 = 21
    GrabRounding                = 22
    ImageBorderSize             = 23
    TabRounding                 = 24
    TabBorderSize               = 25
    TabMinWidthBase             = 26
    TabMinWidthShrink           = 27
    TabBarBorderSize            = 28
    TabBarOverlineSize          = 29
    TableAngledHeadersAngle     = 30
    TableAngledHeadersTextAlign = 31
    TreeLinesSize               = 32
    TreeLinesRounding           = 33
    ButtonTextAlign             = 34
    SelectableTextAlign         = 35
    SeparatorTextBorderSize     = 36
    SeparatorTextAlign          = 37
    SeparatorTextPadding        = 38
    DockingSeparatorSize        = 39


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. Theme — 编辑器主题配置 / Editor Theme Configuration                 ║
# ║                                                                         ║
# ║  修改以下值即可改变整个编辑器外观。所有颜色为线性空间 RGBA。                  ║
# ║  Modify values below to restyle the entire editor.                      ║
# ║  All colors are linear-space RGBA.                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Theme:
    """
    InfEngine 编辑器主题 — 所有颜色、尺寸、图标常量的唯一来源。
    Central theme for the InfEngine Editor — single source of truth for
    all colors, sizes, icons, and layout constants.

    修改此类中的值即可定制编辑器外观。
    Modify values in this class to customize the editor appearance.
    """

    # ══════════════════════════════════════════════════════════════════════
    #  基础调色板 / Base Palette (NASA Punk dark theme)
    #  深空黑底 + 电光青色强调 / Deep space blacks + electric cyan accent
    # ══════════════════════════════════════════════════════════════════════

    # -- 文本颜色 / Text Colors ------------------------------------------------
    TEXT              : RGBA = (0.84,  0.84,  0.87,  1.0)   # 主文本 / Primary text (subtle blue tint)
    TEXT_DISABLED     : RGBA = (0.40,  0.40,  0.44,  1.0)   # 禁用文本 / Disabled text
    TEXT_DIM          : RGBA = (0.55,  0.55,  0.58,  1.0)   # 次要文本 / Secondary/dim text

    # -- 背景颜色 / Background Colors -------------------------------------------
    WINDOW_BG         : RGBA = srgb3(0.10, 0.10, 0.13)      # 窗口背景 / Window background (blue-tinted space black)
    CHILD_BG          : RGBA = (0.0, 0.0, 0.0, 0.0)         # 子窗口背景 / Child window bg (transparent)
    POPUP_BG          : RGBA = srgb3(0.11, 0.11, 0.14, 0.96) # 弹出窗口背景 / Popup background
    STATUS_BAR_BG     : RGBA = srgb3(0.06, 0.06, 0.08)      # 状态栏背景 / Status bar background (deep space)

    # -- 边框颜色 / Border Colors ------------------------------------------------
    BORDER            : RGBA = srgb3(0.17, 0.17, 0.22)      # 标准边框 / Standard border (steel-blue)
    BORDER_TRANSPARENT: RGBA = (0.0, 0.0, 0.0, 0.0)         # 透明边框 / Transparent border
    BORDER_SHADOW     : RGBA = (0.0, 0.0, 0.0, 0.0)         # 边框阴影 / Border shadow

    # -- 输入框/滑块背景 / Frame Colors (input fields, sliders) ------------------
    FRAME_BG          : RGBA = srgb3(0.12, 0.12, 0.16)      # 默认背景 / Default background (blue-tinted)
    FRAME_BG_HOVERED  : RGBA = srgb3(0.15, 0.15, 0.19)      # 悬停时 / On hover
    FRAME_BG_ACTIVE   : RGBA = srgb3(0.13, 0.18, 0.21)      # 激活时 / On active (cyan tint)

    # ══════════════════════════════════════════════════════════════════════
    #  按钮颜色 / Button Colors
    # ══════════════════════════════════════════════════════════════════════

    # -- 标准按钮 / Regular Button -----------------------------------------------
    BTN_NORMAL        : RGBA = srgb3(0.14, 0.14, 0.18)      # 默认 / Normal (steel dark)
    BTN_HOVERED       : RGBA = srgb3(0.16, 0.20, 0.23)      # 悬停 / Hovered (cyan tint)
    BTN_ACTIVE        : RGBA = hex_to_linear(0x00, 0xBC, 0xD4) # 按下 / Active (electric cyan)

    # -- 幽灵按钮 / Ghost Button (透明背景，用于工具栏和状态栏)
    #    Transparent background, used in toolbar and status bar
    BTN_GHOST         : RGBA = (0.0, 0.0, 0.0, 0.0)         # 透明 / Transparent
    BTN_GHOST_HOVERED : RGBA = srgb3(0.14, 0.18, 0.21)      # 悬停 / Hovered (cyan warmth)
    BTN_GHOST_ACTIVE  : RGBA = srgb3(0.16, 0.21, 0.24)      # 按下 / Active (deeper cyan)

    # -- 状态栏幽灵按钮 / Status Bar Ghost Button (略有不同的悬停色)
    BTN_SB_HOVERED    : RGBA = srgb3(0.12, 0.16, 0.18)      # 悬停 / Hovered (subtle cyan)
    BTN_SB_ACTIVE     : RGBA = srgb3(0.14, 0.18, 0.21)      # 按下 / Active

    # -- 选中高亮 / Selection Highlight (用于项目面板等)
    BTN_SELECTED      : RGBA = srgb3(0.14, 0.20, 0.24)      # 选中态 / Selected state (cyan highlight)
    BTN_SUBTLE_HOVER  : RGBA = (0.12, 0.14, 0.17, 1.0)      # 微妙悬停 / Subtle hover on icons

    # -- 工具栏播放按钮 / Toolbar Play-Mode Buttons
    PLAY_ACTIVE       : RGBA = (0.20, 0.45, 0.30, 1.0)      # 播放中(绿色) / Playing (green tint)
    PAUSE_ACTIVE      : RGBA = (0.50, 0.40, 0.15, 1.0)      # 暂停中(琥珀) / Paused (amber tint)
    BTN_IDLE          : RGBA = (0.12, 0.12, 0.15, 1.0)   # 空闲态 / Idle state (steel dark)
    BTN_DISABLED      : RGBA = (0.10, 0.10, 0.10, 0.4)      # 禁用态 / Disabled state

    # -- 强调按钮 / Accent Button
    APPLY_BUTTON      : RGBA = hex_to_linear(0x00, 0xD4, 0xAA) # 应用/确认(青绿) / Apply/Confirm (teal)

    # ══════════════════════════════════════════════════════════════════════
    #  头部/树节点/可选项 / Headers, Tree Nodes, Selectables
    # ══════════════════════════════════════════════════════════════════════

    HEADER            : RGBA = srgb3(0.12, 0.12, 0.16)      # 默认 / Normal
    HEADER_HOVERED    : RGBA = srgb3(0.15, 0.19, 0.22)      # 悬停 / Hovered (cyan tint)
    HEADER_ACTIVE     : RGBA = srgb3(0.16, 0.21, 0.24)      # 按下 / Active (deeper cyan)
    SELECTION_BG      : RGBA = srgb3(0.13, 0.19, 0.23)      # 选择背景 / Selection bg (cyan)

    # ══════════════════════════════════════════════════════════════════════
    #  分割条 / Splitter Colors
    # ══════════════════════════════════════════════════════════════════════

    SPLITTER_HOVER    : RGBA = (0.18, 0.28, 0.33, 0.6)      # 悬停 / Hovered (cyan tint)
    SPLITTER_ACTIVE   : RGBA = (0.18, 0.28, 0.33, 0.8)      # 激活 / Active

    # ══════════════════════════════════════════════════════════════════════
    #  拖放 / Drag & Drop
    # ══════════════════════════════════════════════════════════════════════

    DRAG_DROP_TARGET        : RGBA = (0.0, 0.0, 0.0, 0.0)   # 拖放目标 / Drop target highlight
    DND_DROP_OUTLINE        : RGBA = (1.0, 1.0, 1.0, 0.85)  # 拖放轮廓 / Drop outline color
    DND_DROP_OUTLINE_THICKNESS: float = 1.5                  # 拖放轮廓粗细 / Outline thickness (px)
    DND_REORDER_LINE        : RGBA = (1.0, 1.0, 1.0, 0.90)  # 重排指示线 / Reorder indicator line
    DND_REORDER_LINE_THICKNESS: float = 2.0                  # 重排线粗细 / Line thickness (px)
    DND_REORDER_SEPARATOR_H : float = 6.0                    # 重排分隔器高度 / Separator height (px)

    # ══════════════════════════════════════════════════════════════════════
    #  控制台/日志 / Console & Log Colors
    # ══════════════════════════════════════════════════════════════════════

    LOG_INFO          : RGBA = (0.82,  0.82,  0.85,  1.0)   # 信息日志 / Info log
    LOG_WARNING       : RGBA = (0.890, 0.710, 0.300, 1.0)   # 警告日志 / Warning log (yellow)
    LOG_ERROR         : RGBA = (0.922, 0.341, 0.341, 1.0)   # 错误日志 / Error log (red)
    LOG_TRACE         : RGBA = (0.50,  0.50,  0.50,  1.0)   # 追踪日志 / Trace log (gray)
    LOG_BADGE         : RGBA = (0.55,  0.55,  0.55,  1.0)   # 日志徽章 / Log badge (count)
    LOG_DIM           : RGBA = (0.133, 0.133, 0.133, 0.6)   # 日志暗色 / Dimmed log row

    META_TEXT         : RGBA = (1.0, 1.0, 1.0, 1.0)         # 元信息文本 / Meta text (white)
    SUCCESS_TEXT      : RGBA = (0.70,  0.80,  0.70,  1.0)   # 成功文本 / Success text (green)
    WARNING_TEXT      : RGBA = (0.90,  0.60,  0.20,  1.0)   # 警告文本 / Warning text (orange)
    ERROR_TEXT        : RGBA = (0.90,  0.30,  0.30,  1.0)   # 错误文本 / Error text (red)
    PREFAB_TEXT       : RGBA = (0.42,  0.65,  0.95,  1.0)   # 预制体文本 / Prefab instance text (blue)

    # -- 控制台交替行背景 / Console Alternating Row Background
    ROW_ALT           : RGBA = (0.06, 0.06, 0.09, 0.40)     # 交替行 / Alternate row bg (blue tint)
    ROW_NONE          : RGBA = (0.0,  0.0,  0.0,  0.0)      # 无背景 / No bg

    # ══════════════════════════════════════════════════════════════════════
    #  播放模式边框 / Play-Mode Viewport Border
    # ══════════════════════════════════════════════════════════════════════

    BORDER_PLAY       : RGBA = hex_to_linear(0x03, 0xDE, 0x6D) # 播放中边框(绿) / Playing border (green)
    BORDER_PAUSE      : RGBA = hex_to_linear(0xFF, 0xB7, 0x4D) # 暂停中边框(琥珀) / Paused border (amber)
    BORDER_THICKNESS  : float = 2.0                             # 边框粗细 / Border thickness (px)

    # ══════════════════════════════════════════════════════════════════════
    #  检视器面板 / Inspector Panel Layout & Colors
    # ══════════════════════════════════════════════════════════════════════

    # -- 布局尺寸 / Layout Sizes
    INSPECTOR_INIT_SIZE        = (300, 500)     # 初始窗口大小 / Initial window size (w, h)
    INSPECTOR_MIN_PROPS_H      = 100            # 属性区最小高度 / Min properties height
    INSPECTOR_MIN_RAWDATA_H    = 100            # 原始数据区最小高度 / Min raw-data height
    INSPECTOR_SPLITTER_H       = 8              # 分割条高度 / Splitter bar height
    INSPECTOR_DEFAULT_RATIO    = 0.4            # 属性/原始数据比例 / Properties ratio
    INSPECTOR_LABEL_PAD        = 18.0           # 标签内边距 / Label padding
    INSPECTOR_MIN_LABEL_WIDTH  = 156.0          # 最小标签宽度 / Min label width
    INSPECTOR_FRAME_PAD        = (4.0, 2.0)     # 帧内边距 / Frame padding
    INSPECTOR_ITEM_SPC         = (4.0, 2.0)     # 项间距 / Item spacing
    INSPECTOR_SUBITEM_SPC      = (4.0, 2.0)     # 子项间距 / Sub-item spacing
    INSPECTOR_SECTION_GAP      = 6.0            # 分区间隔 / Section gap
    INSPECTOR_TITLE_GAP        = 10.0           # 标题间隔 / Title gap

    # -- 组件头部 / Component Header
    INSPECTOR_HEADER_PRIMARY_FRAME_PAD = (4.0, 2.0)   # 主头部帧内边距 / Primary header frame padding
    INSPECTOR_HEADER_SECONDARY_FRAME_PAD = (4.0, 2.0) # 次头部帧内边距 / Secondary header frame padding
    INSPECTOR_HEADER_PRIMARY_FONT_SCALE= 1.0           # 主头部字体缩放 / Primary header font scale
    INSPECTOR_HEADER_SECONDARY_FONT_SCALE= 1.0         # 次头部字体缩放 / Secondary header font scale
    INSPECTOR_HEADER_ITEM_SPC   = (4.0, 2.0)           # 头部项间距 / Header item spacing
    INSPECTOR_HEADER_BORDER_SIZE = 0.0                  # 头部边框粗细 / Header border size
    INSPECTOR_ACTION_ALIGN_X    = 0.0                   # 操作按钮对齐 / Action button alignment
    INSPECTOR_HEADER_CONTENT_INDENT = 28.0              # 头部内容缩进 / Header content indent (px)
    ADD_COMP_SEARCH_W          = 240                    # 搜索框宽度 / "Search components" input width
    COMPONENT_ICON_SIZE        = 16                     # 组件图标大小 / Component icon size (px)
    COMP_ENABLED_CB_OFFSET     = 40                     # 启用复选框右偏移 / Enabled checkbox right offset

    # -- 复选框样式 / Checkbox Style
    INSPECTOR_CHECKBOX_FONT_SCALE= 1.0                  # 复选框字体缩放 / Checkbox font scale
    INSPECTOR_CHECKBOX_FRAME_PAD = (4.0, 2.0)           # 复选框帧内边距 / Checkbox frame padding
    INSPECTOR_CHECKBOX_SLOT_W    = 22.0                  # 复选框插槽宽度 / Checkbox slot width

    # -- 检视器头部颜色 / Inspector Header Colors
    INSPECTOR_HEADER_PRIMARY    : RGBA = srgb3(0.14, 0.14, 0.18)     # 主头部 / Primary (steel)
    INSPECTOR_HEADER_PRIMARY_HOVERED : RGBA = srgb3(0.16, 0.20, 0.23) # 主头部悬停 / Primary hovered (cyan)
    INSPECTOR_HEADER_PRIMARY_ACTIVE  : RGBA = srgb3(0.17, 0.22, 0.26) # 主头部激活 / Primary active
    INSPECTOR_HEADER_SECONDARY  : RGBA = srgb3(0.11, 0.11, 0.14)     # 次头部 / Secondary
    INSPECTOR_HEADER_SECONDARY_HOVERED : RGBA = srgb3(0.13, 0.16, 0.19)
    INSPECTOR_HEADER_SECONDARY_ACTIVE  : RGBA = srgb3(0.14, 0.18, 0.21)

    # -- 检视器内联按钮 / Inspector Inline Buttons
    INSPECTOR_INLINE_BTN_IDLE  : RGBA = srgb3(0.12, 0.12, 0.16)     # 空闲 / Idle
    INSPECTOR_INLINE_BTN_HOVER : RGBA = srgb3(0.15, 0.19, 0.22)     # 悬停 / Hover (cyan)
    INSPECTOR_INLINE_BTN_ACTIVE: RGBA = hex_to_linear(0x00, 0xBC, 0xD4) # 按下 / Active (electric cyan)
    INSPECTOR_INLINE_BTN_ON    : RGBA = hex_to_linear(0x00, 0xD4, 0xAA) # 选中 / Active (bright teal)
    INSPECTOR_INLINE_BTN_GAP   : float = 4.0                         # 按钮间距 / Button gap
    INSPECTOR_INLINE_BTN_H     : float = 0.0                         # 按钮高度 / Button height (0=auto)

    # -- 颜色色板边框 / Color Swatch Border
    COLOR_SWATCH_BORDER       : RGBA = (0.4, 0.4, 0.4, 1.0)

    # ══════════════════════════════════════════════════════════════════════
    #  工具栏面板 / Toolbar Panel Spacing
    # ══════════════════════════════════════════════════════════════════════

    TOOLBAR_WIN_PAD   = (4.0, 4.0)     # 窗口内边距 / Window padding
    TOOLBAR_FRAME_PAD = (6.0, 4.0)     # 帧内边距 / Frame padding
    TOOLBAR_ITEM_SPC  = (6.0, 4.0)     # 项间距 / Item spacing
    TOOLBAR_FRAME_RND = 0.0            # 帧圆角 / Frame rounding
    TOOLBAR_FRAME_BRD = 0.0            # 帧边框 / Frame border size

    # ══════════════════════════════════════════════════════════════════════
    #  弹出窗口 / Popup Spacing (用于 Gizmos/Camera 等下拉)
    # ══════════════════════════════════════════════════════════════════════

    POPUP_WIN_PAD     = (16.0, 12.0)   # 弹出窗口内边距 / Popup window padding
    POPUP_ITEM_SPC    = (10.0, 8.0)    # 弹出项间距 / Popup item spacing
    POPUP_FRAME_PAD   = (8.0, 6.0)     # 弹出帧内边距 / Popup frame padding

    # -- 添加组件弹窗 / Add Component Popup
    POPUP_ADD_COMP_PAD  = (10.0, 8.0)  # 内边距 / Padding
    POPUP_ADD_COMP_SPC  = (6.0, 4.0)   # 项间距 / Item spacing
    ADD_COMP_FRAME_PAD  = (6.0, 6.0)   # 帧内边距 / Frame padding

    # ══════════════════════════════════════════════════════════════════════
    #  层级面板 / Hierarchy Panel
    # ══════════════════════════════════════════════════════════════════════

    TREE_ITEM_SPC     = (0.0, 3.0)     # 树节点项间距 / Tree item spacing
    TREE_FRAME_PAD    = (4.0, 5.0)     # 树节点帧内边距 / Tree frame padding (makes nodes taller)

    # ══════════════════════════════════════════════════════════════════════
    #  控制台面板 / Console Panel Spacing
    # ══════════════════════════════════════════════════════════════════════

    CONSOLE_FRAME_PAD = (4.0, 3.0)     # 帧内边距 / Frame padding
    CONSOLE_ITEM_SPC  = (6.0, 4.0)     # 项间距 / Item spacing

    # ══════════════════════════════════════════════════════════════════════
    #  状态栏 / Status Bar Layout
    # ══════════════════════════════════════════════════════════════════════

    STATUS_BAR_WIN_PAD   = (6.0, 4.0)   # 窗口内边距 / Window padding
    STATUS_BAR_ITEM_SPC  = (8.0, 0.0)   # 项间距 / Item spacing
    STATUS_BAR_FRAME_PAD = (0.0, 0.0)   # 帧内边距 / Frame padding

    # -- 状态/进度指示器 / Status/Progress Indicator (状态栏右侧)
    STATUS_PROGRESS_FRACTION : float = 0.20                           # 右侧占比 / Right fraction
    STATUS_PROGRESS_H        : float = 4.0                            # 进度条高度 / Progress bar height
    STATUS_PROGRESS_CLR      : RGBA = (0.00, 0.60, 0.70, 1.0)       # 进度色(青) / Progress color (teal)
    STATUS_PROGRESS_BG       : RGBA = (0.10, 0.10, 0.10, 1.0)       # 进度背景 / Progress bg
    STATUS_PROGRESS_LABEL_CLR: RGBA = (0.65, 0.65, 0.65, 1.0)       # 进度文本 / Progress label color

    # ══════════════════════════════════════════════════════════════════════
    #  项目面板 / Project Panel
    # ══════════════════════════════════════════════════════════════════════

    ICON_BTN_NO_PAD   = (0.0, 0.0)     # 图标按钮内边距 / Icon button frame padding (none)
    PROJECT_PANEL_PAD = (12.0, 8.0)    # 文件网格内边距 / File grid child window padding

    # ══════════════════════════════════════════════════════════════════════
    #  场景视图面板 / Scene View Panel
    # ══════════════════════════════════════════════════════════════════════

    # -- Gizmo 工具按钮 / Gizmo Tool Buttons
    SCENE_GIZMO_TOOL_BTN_W    : float = 20.0     # 按钮宽度 / Button width
    SCENE_GIZMO_TOOL_BTN_H    : float = 20.0     # 按钮高度 / Button height
    SCENE_GIZMO_TOOL_BTN_GAP  : float = 1.0      # 按钮间距 / Button gap
    SCENE_GIZMO_TOOL_BTN_PAD  = (2.0, 2.0)       # 帧内边距 / Frame padding
    SCENE_COORD_DROPDOWN_W    : float = 80.0      # 坐标系下拉宽度 / Global/Local dropdown width

    # -- 朝向 Gizmo / Orientation Gizmo (右上角坐标轴小部件)
    SCENE_ORIENT_RADIUS       : float = 40.0      # 圆半径 / Circle radius
    SCENE_ORIENT_MARGIN       : float = 12.0      # 边距 / Margin from corner
    SCENE_ORIENT_AXIS_LEN     : float = 30.0      # 轴线长度 / Axis line length
    SCENE_ORIENT_END_RADIUS   : float = 7.0       # 轴端圆半径 / Axis end circle radius
    SCENE_ORIENT_NEG_RADIUS   : float = 4.0       # 负轴端圆半径 / Negative axis circle radius
    SCENE_ORIENT_BG           : RGBA = (0.10, 0.10, 0.14, 0.6)   # 背景色 / Background (space dark)
    SCENE_ORIENT_FLY_DURATION : float = 0.3       # 飞行动画时长(秒) / Fly animation duration (s)

    # -- 场景覆盖层下拉 / Scene Overlay Dropdown
    SCENE_OVERLAY_COMBO_BG    : RGBA = (0.10, 0.10, 0.14, 0.85)  # 背景 / Background (space dark)
    SCENE_OVERLAY_COMBO_HOVER : RGBA = (0.16, 0.20, 0.24, 0.90)  # 悬停 / Hover (cyan tint)
    SCENE_OVERLAY_COMBO_ACTIVE: RGBA = (0.13, 0.17, 0.20, 0.95)  # 激活 / Active
    SCENE_OVERLAY_ROUNDING    : float = 4.0       # 圆角 / Rounding
    SCENE_OVERLAY_BORDER_SIZE : float = 0.0       # 边框 / Border size

    # ══════════════════════════════════════════════════════════════════════
    #  UI 编辑器面板 / UI Editor Panel
    # ══════════════════════════════════════════════════════════════════════

    # -- 画布 / Canvas
    UI_EDITOR_CANVAS_BG       : RGBA = (0.10, 0.10, 0.13, 1.0)   # 画布背景 / Canvas background (space dark)
    UI_EDITOR_CANVAS_BORDER   : RGBA = (0.30, 0.35, 0.40, 1.0)   # 画布边框 / Canvas border (steel)

    # -- 元素交互 / Element Interaction
    UI_EDITOR_ELEMENT_HOVER   : RGBA = (0.00, 0.74, 0.83, 0.12)  # 元素悬停 / Element hover (cyan glow)
    UI_EDITOR_ELEMENT_SELECT  : RGBA = (0.00, 0.74, 0.83, 1.0)   # 元素选中 / Element selected (electric cyan)

    # -- 控制柄 / Handles
    UI_EDITOR_HANDLE_COLOR    : RGBA = (1.0, 1.0, 1.0, 1.0)      # 控制柄颜色 / Handle color
    UI_EDITOR_HANDLE_SIZE     : float = 4.0                        # 控制柄半径 / Handle half-size (px)

    # -- 缩放与视口 / Zoom & Viewport
    UI_EDITOR_TOOLBAR_HEIGHT  : float = 32.0      # 工具栏高度 / Toolbar height
    UI_EDITOR_MIN_ZOOM        : float = 0.05      # 最小缩放 / Min zoom
    UI_EDITOR_MAX_ZOOM        : float = 2.0       # 最大缩放(200%) / Max zoom (200%)
    UI_EDITOR_ZOOM_STEP       : float = 0.1       # 滚轮缩放步长 / Wheel zoom step

    # -- 标签 / Labels
    UI_EDITOR_LABEL_OFFSET    : float = 16.0      # 画布顶部标签偏移 / Canvas top label offset (px)
    UI_EDITOR_LABEL_COLOR     : RGBA = (0.6, 0.6, 0.6, 0.7)      # 标签颜色 / Label color

    # -- 旋转控制柄 / Rotation Handle
    UI_EDITOR_ROTATE_DISTANCE : float = 22.0      # 旋转柄偏移 / Offset from top-mid (px)
    UI_EDITOR_ROTATE_RADIUS   : float = 4.0       # 旋转柄半径 / Circle radius (px)
    UI_EDITOR_ROTATE_HIT_R    : float = 10.0      # 旋转柄点击半径 / Click radius (px)
    UI_EDITOR_EDGE_HIT_TOL    : float = 6.0       # 边缘点击容差 / Edge hit tolerance (px)
    UI_EDITOR_SELECT_LINE_W   : float = 1.5       # 选择边框线宽 / Selection border width
    UI_EDITOR_ROTATE_LINE_W   : float = 1.0       # 旋转柄线宽 / Rotate handle line width
    UI_EDITOR_MIN_ELEM_SIZE   : float = 4.0       # 最小元素尺寸 / Min element dimension (px)

    # -- 占位符 / Placeholder
    UI_EDITOR_PLACEHOLDER_TINT : float = 0.3      # 占位符色调 / Placeholder tint multiplier
    UI_EDITOR_PLACEHOLDER_ALPHA: float = 0.5      # 占位符透明度 / Placeholder alpha
    UI_EDITOR_FALLBACK_TEXT   : RGBA = (0.7, 0.7, 0.7, 1.0)      # 回退文本色 / Fallback text color

    # -- 窗口/工具栏布局 / Window & Toolbar Layout
    UI_EDITOR_INIT_WINDOW_W   : float = 800.0     # 初始窗口宽度 / Initial window width
    UI_EDITOR_INIT_WINDOW_H   : float = 600.0     # 初始窗口高度 / Initial window height
    UI_EDITOR_FIT_MARGIN      : float = 40.0      # 适应缩放边距 / Fit-zoom padding (px)
    UI_EDITOR_TOOLBAR_GAP     : float = 4.0       # 工具栏按钮间距 / Toolbar button gap
    UI_EDITOR_TOOLBAR_SECTION_GAP : float = 16.0  # 工具栏分区间距 / Toolbar section gap
    UI_EDITOR_CREATE_BTN_W    : float = 220.0     # 创建按钮宽度 / "Create Canvas" button width
    UI_EDITOR_CREATE_BTN_H    : float = 28.0      # 创建按钮高度 / "Create Canvas" button height

    # -- 默认创建尺寸 / Default Element Creation Sizes (设计像素)
    UI_EDITOR_NEW_TEXT_POS    = (-80.0, -20.0)
    UI_EDITOR_NEW_IMAGE_SIZE  = (100.0, 100.0)
    UI_EDITOR_NEW_IMAGE_POS   = (-50.0, -50.0)
    UI_EDITOR_NEW_BUTTON_SIZE = (160.0, 40.0)
    UI_EDITOR_NEW_BUTTON_POS  = (-80.0, -20.0)

    # -- 缩放自适应吸附表 / Zoom-Adaptive Snap Table (zoom_threshold → grid_step)
    UI_EDITOR_SNAP_TABLE = (
        (1.0,  1),
        (0.75, 2),
        (0.5,  5),
        (0.35, 10),
        (0.2,  20),
        (0.1,  50),
    )
    UI_EDITOR_SNAP_DEFAULT    : int = 100          # 最远缩放步长 / Step at smallest zoom

    # -- 对齐引导线 / Alignment Guides
    UI_EDITOR_ALIGN_GUIDE     : RGBA = (0.18, 0.72, 1.0, 0.95)   # 引导线色 / Guide line color
    UI_EDITOR_ALIGN_GUIDE_FAINT: RGBA = (0.18, 0.72, 1.0, 0.30)  # 淡引导线色 / Faint guide color
    UI_EDITOR_ALIGN_GUIDE_W   : float = 1.5       # 引导线宽度 / Guide line width
    UI_EDITOR_ALIGN_SNAP_PX   : float = 8.0       # 吸附像素 / Snap threshold (px)
    UI_EDITOR_ALIGN_BTN_W     : float = 34.0      # 对齐按钮宽度 / Align button width
    UI_EDITOR_ALIGN_BTN_H     : float = 24.0      # 对齐按钮高度 / Align button height
    UI_EDITOR_ALIGN_BTN_GAP   : float = 4.0       # 对齐按钮间距 / Align button gap

    # ══════════════════════════════════════════════════════════════════════
    #  UI 运行时默认值 / UI Runtime Defaults (InfScreenUI 渲染)
    # ══════════════════════════════════════════════════════════════════════

    UI_DEFAULT_BUTTON_BG      : RGBA = (0.22, 0.56, 0.92, 1.0)   # 默认按钮背景 / Default button bg
    UI_DEFAULT_LABEL_COLOR    : RGBA = (1.0, 1.0, 1.0, 1.0)      # 默认标签色 / Default label color
    UI_DEFAULT_FONT_SIZE      : float = 20.0                       # 默认字号 / Default font size
    UI_DEFAULT_LINE_HEIGHT    : float = 1.2                        # 默认行高 / Default line height
    UI_DEFAULT_LETTER_SPACING : float = 0.0                        # 默认字间距 / Default letter spacing

    # ══════════════════════════════════════════════════════════════════════
    #  构建设置面板 / Build Settings Panel
    # ══════════════════════════════════════════════════════════════════════

    BUILD_SETTINGS_ROW_SPC = (4.0, 6.0)   # 行间距 / Row spacing

    # ══════════════════════════════════════════════════════════════════════
    #  常用窗口标志组合 / Common Window Flag Combos
    # ══════════════════════════════════════════════════════════════════════

    WINDOW_FLAGS_VIEWPORT  = (ImGuiWindowFlags.NoFocusOnAppearing
                              | ImGuiWindowFlags.NoBringToFrontOnFocus)
    WINDOW_FLAGS_NO_SCROLL = (ImGuiWindowFlags.NoScrollbar
                              | ImGuiWindowFlags.NoScrollWithMouse)
    WINDOW_FLAGS_NO_DECOR  = (ImGuiWindowFlags.NoTitleBar
                              | ImGuiWindowFlags.NoResize
                              | ImGuiWindowFlags.NoMove
                              | ImGuiWindowFlags.NoScrollbar
                              | ImGuiWindowFlags.NoScrollWithMouse
                              | ImGuiWindowFlags.NoSavedSettings
                              | ImGuiWindowFlags.NoFocusOnAppearing
                              | ImGuiWindowFlags.NoDocking
                              | ImGuiWindowFlags.NoInputs)
    WINDOW_FLAGS_FLOATING  = (ImGuiWindowFlags.NoCollapse
                              | ImGuiWindowFlags.NoSavedSettings)
    WINDOW_FLAGS_DIALOG    = (ImGuiWindowFlags.NoCollapse
                              | ImGuiWindowFlags.NoSavedSettings
                              | ImGuiWindowFlags.NoDocking
                              | ImGuiWindowFlags.NoResize
                              | ImGuiWindowFlags.NoMove)

    # ══════════════════════════════════════════════════════════════════════
    #  ImGui 条件常量 / ImGui Condition Constants
    # ══════════════════════════════════════════════════════════════════════

    COND_FIRST_USE_EVER = 4   # 仅在首次使用时设置 / Set only on first use
    COND_ALWAYS         = 1   # 每帧都设置 / Set every frame

    # ══════════════════════════════════════════════════════════════════════
    #  边框尺寸 / Border Sizes
    # ══════════════════════════════════════════════════════════════════════

    BORDER_SIZE_NONE    = 0.0   # 无边框 / No border

    # ══════════════════════════════════════════════════════════════════════
    #  图标常量 / Icon Constants
    #  文本图标用于没有图片图标的地方；图片图标名用于 EditorIcons.get()
    #  Text icons for fallback; image icon names for EditorIcons.get()
    # ══════════════════════════════════════════════════════════════════════

    # -- 文本图标 / Text Icons (Unicode 符号)
    ICON_PLUS          : str = "+"
    ICON_MINUS         : str = "-"
    ICON_REMOVE        : str = "\u00d7"         # × 乘号 / Multiplication sign
    ICON_PICKER        : str = "\u2299"         # ⊙ 目标 / Circled dot
    ICON_WARNING       : str = "\u25b2"         # ▲ 三角 / Triangle
    ICON_ERROR         : str = "\u25cf"         # ● 圆点 / Filled circle
    ICON_DOT           : str = "\u00b7"         # · 中点 / Middle dot
    ICON_CHECK         : str = "\u2713"         # ✓ 对勾 / Check mark

    # -- 图片图标名 / Image Icon Names (由 EditorIcons.get 加载)
    ICON_IMG_PLUS      : str = "plus"
    ICON_IMG_MINUS     : str = "minus"
    ICON_IMG_REMOVE    : str = "remove"
    ICON_IMG_PICKER    : str = "picker"
    ICON_IMG_WARNING   : str = "warning"
    ICON_IMG_ERROR     : str = "error"
    ICON_IMG_UI_TEXT   : str = "ui_text"
    ICON_IMG_UI_IMAGE  : str = "ui_image"
    ICON_IMG_UI_BUTTON : str = "ui_button"
    EDITOR_ICON_SIZE   : float = 16.0           # 默认图标大小 / Default icon size (px)

    # ══════════════════════════════════════════════════════════════════════
    #  4. 样式推入/弹出辅助方法 / Style Push/Pop Helpers
    #     这些方法将多个 push 操作封装为一次调用。
    #     These methods bundle multiple push operations into single calls.
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def push_ghost_button_style(ctx) -> int:
        """推入透明按钮样式。返回推入的颜色数量 (3)。
        Push transparent button colors. Returns color count pushed (3)."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_GHOST_HOVERED)
        ctx.push_style_color(ImGuiCol.ButtonActive,   *Theme.BTN_GHOST_ACTIVE)
        return 3

    @staticmethod
    def push_flat_button_style(ctx, r: float, g: float, b: float, a: float = 1.0) -> int:
        """推入纯色按钮样式（自动生成高亮/按下色）。返回 3。
        Push flat solid-color button style. Returns 3."""
        ctx.push_style_color(ImGuiCol.Button,        r, g, b, a)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  min(r + 0.06, 1), min(g + 0.06, 1), min(b + 0.06, 1), a)
        ctx.push_style_color(ImGuiCol.ButtonActive,   min(r + 0.12, 1), min(g + 0.12, 1), min(b + 0.12, 1), a)
        return 3

    @staticmethod
    def push_toolbar_vars(ctx) -> int:
        """推入工具栏紧凑间距预设。返回 var 数量 (5)。
        Push compact toolbar spacing preset. Returns var count (5)."""
        ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, *Theme.TOOLBAR_WIN_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding,  *Theme.TOOLBAR_FRAME_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing,   *Theme.TOOLBAR_ITEM_SPC)
        ctx.push_style_var_float(ImGuiStyleVar.FrameRounding, Theme.TOOLBAR_FRAME_RND)
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, Theme.TOOLBAR_FRAME_BRD)
        return 5

    @staticmethod
    def push_popup_vars(ctx) -> int:
        """推入弹出窗口间距预设。返回 3。
        Push popup spacing preset. Returns 3."""
        ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, *Theme.POPUP_WIN_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing,   *Theme.POPUP_ITEM_SPC)
        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding,  *Theme.POPUP_FRAME_PAD)
        return 3

    @staticmethod
    def push_status_bar_button_style(ctx) -> int:
        """推入状态栏按钮样式。返回 3。
        Push status bar button style. Returns 3."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_SB_HOVERED)
        ctx.push_style_color(ImGuiCol.ButtonActive,   *Theme.BTN_SB_ACTIVE)
        return 3

    @staticmethod
    def push_transparent_border(ctx) -> int:
        """推入透明边框颜色。返回 1。
        Push transparent border color. Returns 1."""
        ctx.push_style_color(ImGuiCol.Border, *Theme.BORDER_TRANSPARENT)
        return 1

    @staticmethod
    def push_drag_drop_target_style(ctx) -> int:
        """推入拖放目标高亮色。返回 1。
        Push drag-drop target highlight color. Returns 1."""
        ctx.push_style_color(ImGuiCol.DragDropTarget, *Theme.DRAG_DROP_TARGET)
        return 1

    @staticmethod
    def push_console_toolbar_vars(ctx) -> int:
        """推入控制台工具栏紧凑间距。返回 3。
        Push console toolbar compact spacing. Returns 3."""
        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.CONSOLE_FRAME_PAD)
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing,  *Theme.CONSOLE_ITEM_SPC)
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, Theme.TOOLBAR_FRAME_BRD)
        return 3

    @staticmethod
    def push_splitter_style(ctx) -> int:
        """推入分割条样式。返回 3。
        Push splitter button style. Returns 3."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.SPLITTER_HOVER)
        ctx.push_style_color(ImGuiCol.ButtonActive,   *Theme.SPLITTER_ACTIVE)
        return 3

    @staticmethod
    def push_selected_icon_style(ctx) -> int:
        """推入选中图标按钮的高亮色。返回 2。
        Push selected icon button highlight. Returns 2."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_SELECTED)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_SELECTED)
        return 2

    @staticmethod
    def push_unselected_icon_style(ctx) -> int:
        """推入未选中图标按钮样式。返回颜色数 2 (调用者还需 pop 1 var)。
        Push unselected icon button style. Returns 2 colors (caller also pops 1 var)."""
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, 0.0)
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_SUBTLE_HOVER)
        return 2

    @staticmethod
    def get_play_border_color(is_paused: bool) -> RGBA:
        """获取播放模式边框颜色。
        Return the play-mode border color."""
        return Theme.BORDER_PAUSE if is_paused else Theme.BORDER_PLAY

    @staticmethod
    def push_inline_button_style(ctx, active: bool = False) -> int:
        """推入内联按钮样式。返回颜色数 (3)。
        Push inline button style. Returns color count (3)."""
        if active:
            ctx.push_style_color(ImGuiCol.Button, *Theme.INSPECTOR_INLINE_BTN_ON)
            ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.INSPECTOR_INLINE_BTN_ON)
            ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.INSPECTOR_INLINE_BTN_ACTIVE)
        else:
            ctx.push_style_color(ImGuiCol.Button, *Theme.INSPECTOR_INLINE_BTN_IDLE)
            ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.INSPECTOR_INLINE_BTN_HOVER)
            ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.INSPECTOR_INLINE_BTN_ACTIVE)
        return 3

    @staticmethod
    def render_inline_button_row(
        ctx,
        row_id: str,
        items: Iterable[tuple[str, str]],
        *,
        active_items: Optional[Iterable[str]] = None,
        height: float = 0.0,
    ):
        """渲染一行均匀分布的按钮，返回被点击的按钮 id。
        Render a row of evenly sized buttons. Returns clicked item id."""
        entries = list(items)
        if not entries:
            return None

        active_set = set(active_items or [])
        spacing = Theme.INSPECTOR_INLINE_BTN_GAP
        button_h = height if height > 0.0 else Theme.INSPECTOR_INLINE_BTN_H
        avail_w = max(0.0, ctx.get_content_region_avail_width())
        total_gap = spacing * max(0, len(entries) - 1)
        button_w = max(1.0, (avail_w - total_gap) / max(1, len(entries)))

        clicked = [None]
        for idx, (item_id, label) in enumerate(entries):
            color_count = Theme.push_inline_button_style(ctx, item_id in active_set)

            def _on_click(iid=item_id):
                clicked[0] = iid

            ctx.button(f"{label}##{row_id}_{item_id}", _on_click, width=button_w, height=button_h)
            ctx.pop_style_color(color_count)
            if idx + 1 < len(entries):
                ctx.same_line(0, spacing)
        return clicked[0]
