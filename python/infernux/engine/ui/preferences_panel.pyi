"""preferences_panel — standalone Preferences window."""

from __future__ import annotations

from infernux.engine.ui.editor_panel import FloatingEditorPanel


class PreferencesPanel(FloatingEditorPanel):
    """User Preferences utility surface."""

    def __init__(self) -> None: ...
    def on_render_content(self, ctx: object) -> None: ...
