# Infernux 编辑器主题模板

主题是普通插件，只交付数据，无需 preload、原生库或 Runtime 脚本。

1. 修改 `package/inx_package.json` 中的 reference、名称和版本。
2. 编辑 `package/editor/themes/studio.theme.json`。id 是包内身份，活动身份为 `<reference>/<id>`；多个文件可提供不同主题，id 必须唯一。
3. 运行 `python package.py build dist/studio-theme.inxpkg`，通过编辑器插件菜单安装。在 **编辑 → 主题…** 中选择，保存数据后沿普通资源刷新生效。
4. 发布时推送与版本一致的 tag，例如 `v0.1.0`。GitHub workflow 生成普通插件归档和 `infernux-plugin-release.json`，上传到 Release。需要供其他作者使用时，将仓库设为 GitHub Template。

`theme-tokens.json` 从引擎唯一 token 表生成。颜色为 [0,1] 的 sRGB RGBA，vec2 为两个有限数字，float 为一个有限数字；GLOBAL 尺寸不得为负。未知字段、未知 token、错误类型和重复 id 会在导入与打包时报告。优先修改 ROLE_*，再按需覆盖组件 token；省略值继承内置基础表。UI_DEFAULT_* 属于游戏内容，不能由编辑器主题覆盖。

字体继续使用编辑器既有字体和字号设置。主题 JSON 支持颜色、间距、圆角、边框、图标尺寸和以下可选的原生表面样式；不接受字体文件、字号或窗口结构声明。颜色切换不重建字体资源。

| 样式 | 颜色 token | 数值 token |
| --- | --- | --- |
| 面板渐变 | `PANEL_GRADIENT_TOP`、`PANEL_GRADIENT_BOTTOM` | — |
| 面板分层边框 | `PANEL_FRAME_COLOR`、`PANEL_FRAME_INNER_COLOR`、`PANEL_FRAME_HIGHLIGHT`、`PANEL_GLOW_COLOR` | `PANEL_FRAME_WIDTH`、`PANEL_FRAME_INSET`、`PANEL_CORNER_CUT`、`PANEL_GLOW_RADIUS` |
| 按钮渐变及状态 | `BUTTON_GRADIENT_TOP/BOTTOM`、`BUTTON_HOVER_GRADIENT_TOP/BOTTOM`、`BUTTON_ACTIVE_GRADIENT_TOP/BOTTOM` | — |
| 按钮分层边框 | `BUTTON_FRAME_COLOR`、`BUTTON_FRAME_INNER_COLOR`、`BUTTON_FRAME_HIGHLIGHT`、`BUTTON_GLOW_COLOR` | `BUTTON_FRAME_WIDTH`、`BUTTON_FRAME_INSET`、`BUTTON_CORNER_CUT`、`BUTTON_GLOW_RADIUS` |
| 静态纸面纤维 | `SURFACE_GRAIN_TINT` | `SURFACE_GRAIN_OPACITY`、`SURFACE_GRAIN_SCALE` |
| 六边形底纹 | `SURFACE_HEX_TINT` | `SURFACE_HEX_OPACITY`、`SURFACE_HEX_SCALE` |

表中 `TOP/BOTTOM` 代表两个独立 token。新样式默认关闭：渐变端点默认透明，边框宽度和底纹透明度默认为零。启用边框时同时设置颜色与宽度；切角和发光半径以逻辑像素表示，统一跟随 DPI。按钮样式作用于公共按钮控件；其余标准控件继续使用全局颜色、边框和间距。底纹透明度为 [0,1]、比例为 [0.25,8]，每种底纹首次使用时生成一次纹理并缓存，绘制在文字和控件下面。主题不在每帧运行 Python 绘制脚本，也不需要附带纹理文件。

场景工具栏的选中背景为 `SCENE_TOOL_SELECTED_BG/HOVER/ACTIVE`，普通与选中图标为 `SCENE_TOOL_ICON`、`SCENE_TOOL_ICON_SELECTED`；默认从强调色与文字色派生，浅色主题也能保持图标可读。

仅 `package/` 进入安装包。README、CHANGELOG、CI、token 参考、theme_schema.py 和打包器留在仓库外层；不要往 package/ 放缓存、.runtime、预览工程或构建工具。editor/ 内容不进入 Player。

引擎源码仓库的导出命令为 `python scripts/plugins/export_theme_template.py <output-folder>`。导出结果只依赖 Python 标准库，不要求安装引擎。
