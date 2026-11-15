# Infernux editor theme template

This is a normal plugin containing only theme data. No preload class, native
library, runtime script or separate installation mechanism is required.

1. Set your own `reference`, name and version in `package/inx_package.json`.
2. Edit `package/editor/themes/studio.theme.json`. Theme IDs are local to the
   package; the live ID is `<reference>/<id>`. Multiple files provide variants.
3. Run `python package.py build dist/studio-theme.inxpkg` and install the file
   using the editor plugin manager. Open **Edit → Themes** to select it.
4. To publish, push a tag matching the manifest version, such as `v0.1.0`.
   The included GitHub workflow packages the plugin and uploads the archive
   and `infernux-plugin-release.json` to that release. Mark your GitHub repository as a
   template if other authors should start from it.

`theme-tokens.json` lists supported token names, generated from the engine's
single native table. Colors are sRGB RGBA values in [0, 1]; vec2 values contain
two finite numbers; floats contain one finite number. Unknown tokens are import
errors. Start with semantic `ROLE_*` colors, then override individual tokens
only where necessary. Unspecified values use builtin defaults. `UI_DEFAULT_*`
belongs to game UI and cannot be changed by editor themes.

The packager validates the same declaration rules used by the editor, including
metadata types, unknown fields and duplicate IDs. `theme_schema.py` is exported
from the engine's standard-library-only validator; it never imports the engine.
Fonts continue to use the editor's existing font and font-size preferences.
Theme JSON supports colors and layout metrics, not font files, font sizes or
window structure. Color changes do not rebuild font resources.

Optional native surface styling is also available:

- `PANEL_GRADIENT_TOP/BOTTOM` set the two panel gradient endpoints.
- `PANEL_FRAME_COLOR`, `PANEL_FRAME_INNER_COLOR`, `PANEL_FRAME_HIGHLIGHT` and
  `PANEL_GLOW_COLOR` control layered panel edges. Their dimensions are
  `PANEL_FRAME_WIDTH`, `PANEL_FRAME_INSET`, `PANEL_CORNER_CUT` and `PANEL_GLOW_RADIUS`.
- Common button controls use the corresponding `BUTTON_*` tokens, with separate
  `BUTTON_HOVER_GRADIENT_TOP/BOTTOM` and `BUTTON_ACTIVE_GRADIENT_TOP/BOTTOM` colors.
  Other standard controls keep using the global palette, borders and spacing.
- `SURFACE_GRAIN_TINT/OPACITY/SCALE` control static paper fibers;
  `SURFACE_HEX_TINT/OPACITY/SCALE` control the hexagonal pattern.

Slash-separated suffixes denote separate tokens. Fills default to transparent;
frame widths and pattern opacities default to zero. Set both color and width to
enable a frame. Dimensions are logical pixels and scale with DPI. Opacity is in
[0, 1] and pattern scale in [0.25, 8]. Each pattern is generated and uploaded only
on first use, then reused behind editor content. No frame-by-frame Python drawing
or packaged texture files are required.

Scene tool selection uses `SCENE_TOOL_SELECTED_BG/HOVER/ACTIVE`; icon tint uses
`SCENE_TOOL_ICON` and `SCENE_TOOL_ICON_SELECTED`. They derive from the semantic
accent and text roles unless explicitly overridden, including for light themes.

`package/` is the installed payload. This README, the workflow, token reference
and packager stay outside it. Keep a stable package reference and theme ID when
publishing updates. Use the normal `engine` version requirement for compatibility.

The engine source exports this repository with:

```powershell
python scripts/plugins/export_theme_template.py <output-folder>
```
