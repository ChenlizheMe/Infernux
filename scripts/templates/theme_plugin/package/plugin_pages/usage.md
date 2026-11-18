# Studio Theme

Install and enable this package in the normal plugin manager, then open
**Edit → Themes** and select **Studio**. Theme selection is saved as an editor
preference. Disabling the package selects the builtin theme; enabling it again
restores the saved selection.

Edit `editor/themes/studio.theme.json` in the installed package to preview
changes. A malformed theme is removed from the live registry and reported in
Console; saving valid data makes the current theme available again. Package
updates overwrite publisher files, including this theme file. Keep permanent
custom themes under your own package reference.

Editor themes do not change game UI, scene materials or Player rendering.
