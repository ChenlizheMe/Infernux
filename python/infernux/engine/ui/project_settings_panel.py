"""One project settings window, with Player, physics and tag/layer pages."""
from __future__ import annotations

from infernux.engine.project_context import get_project_root
from .build_settings_panel import BuildSettingsPanel, _BUILD_SETTINGS_INTERACTION
from .panel_registry import editor_panel
from .settings_workspace import SettingsWorkspaceMixin, render_settings_workspace
from .tag_layer_settings import PhysicsLayerMatrixPanel, TagLayerSettingsPanel


@editor_panel(
    "Project Settings", type_id="project_settings", title_key="project_settings.title", menu_path="",
    interaction=_BUILD_SETTINGS_INTERACTION,
)
class ProjectSettingsPanel(SettingsWorkspaceMixin, BuildSettingsPanel):
    """Embedded pages share the existing project document and one view identity."""
    SETTINGS_PAGES = (
        ("player", "project_settings.player", "settings.description_build_settings", "player build windows linux android web 玩家 构建 平台"),
        ("physics", "physics.title", "settings.description_physics_settings", "physics gravity collision timestep 物理 重力 碰撞"),
        ("tags_layers", "panel.tags_layers", "settings.description_tag_layer_settings", "tag layer 标签 图层"),
    )

    def __init__(self):
        super().__init__()
        self._settings_page = "player"
        self._settings_pages = {
            "physics": PhysicsLayerMatrixPanel(),
            "tags_layers": TagLayerSettingsPanel(),
        }
        for page in self._settings_pages.values():
            page.set_panel_identity("project_settings", "project_settings")
            page.set_project_path(get_project_root())
            # These controllers render inline, so edits belong to the host
            # view and its selected page, never to a second panel lifecycle.
            page.publish_interaction_ownership = self.publish_interaction_ownership

    def _initial_size(self):
        return 1040.0, 740.0

    def on_enable(self):
        super().on_enable()
        for page in self._settings_pages.values():
            page.on_enable()

    def on_disable(self):
        for page in self._settings_pages.values():
            page.on_disable()
        super().on_disable()

    def on_render_content(self, ctx):
        render_settings_workspace(ctx, self, self._render_page)

    def _render_page(self, ctx, page):
        if page == "player":
            super().on_render_content(ctx)
        else:
            self._settings_pages[page].on_render_content(ctx)
