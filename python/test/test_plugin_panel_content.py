from types import SimpleNamespace

import pytest

from infernux.core.asset_types import TextureType
from infernux.engine.ui.plugin_panel import PluginPanel
import infernux.engine.ui.plugin_panel as panel_module
from infernux.engine import runtime_event_queue


@pytest.mark.parametrize("pages", [[], [{"id": "usage", "title": "Usage"}, {"id": "notes", "title": "Notes"}]])
def test_switching_plugins_selects_first_page_without_locking_it(pages):
    panel = PluginPanel()
    calls = []
    ctx = SimpleNamespace(
        begin_tab_bar=lambda name: True,
        end_tab_bar=lambda: None,
        begin_tab_item=lambda title, selected=False: calls.append((title, selected)) or False,
    )
    manager = SimpleNamespace(content_pages=lambda row: pages)

    def render(reference):
        calls.clear()
        panel._render_detail_pages(ctx, manager, {"reference": reference, "_installed": True}, None)
        return [selected for title, selected in calls]

    assert render("vendor/first") == [True] + [False] * max(1, len(pages))
    assert render("vendor/first") == [False] * (max(1, len(pages)) + 1)
    assert render("vendor/second")[0] is True
    assert render("vendor/first")[0] is True


def test_plugin_panel_filters_by_stable_category_key():
    registry = SimpleNamespace(
        available=lambda: (
            {
                "reference": "infernux/platform-web",
                "name": "Web Platform",
                "category": "platform_build",
                "source": {"official": True},
            },
            {
                "reference": "infernux/mcp",
                "name": "MCP",
                "category": "editor_tools",
                "source": {"official": True},
            },
        ),
        installed_metadata=lambda: (),
    )
    manager = SimpleNamespace(
        registry=registry,
        states={},
        cached_reference_path=lambda _reference: "",
    )
    panel = PluginPanel()
    panel._category_index = 1

    rows = panel._visible_rows(manager)

    assert [row["reference"] for row in rows] == ["infernux/platform-web"]
    assert rows[0]["_category_key"] == "platform_build"


def test_document_images_request_ui_color_and_full_page_resolution(monkeypatch):
    panel = PluginPanel()
    requests = []
    monkeypatch.setattr(panel_module, "_metric", lambda ctx, value: value)
    monkeypatch.setattr(panel_module, "render_resource_preview_rect", lambda *args, **kwargs: requests.append(kwargs) or True)
    ctx = SimpleNamespace(get_content_region_avail_width=lambda: 900)
    manager = SimpleNamespace(content_asset_path=lambda *args: "/downloaded/plugin_pages/overview.png")
    panel._render_markdown_image(ctx, manager, {}, {}, {"source": "overview.png"})
    settings = requests[0]["texture_settings"]
    assert settings.texture_type == TextureType.UI
    assert settings.srgb is True
    assert settings.max_size >= 720
    assert requests[0]["preserve_aspect"] is True


def test_plugin_package_mutation_runs_after_the_render_frame():
    panel = PluginPanel()
    calls = []
    runtime_event_queue.clear()

    panel._run(
        lambda: calls.append("commit") or SimpleNamespace(reference="vendor/plugin"),
        "toggle",
    )

    assert calls == []
    assert panel._pending_action is True
    assert runtime_event_queue.drain() == 1
    assert calls == ["commit"]
    assert panel._pending_action is False
    assert panel._selected_reference == "vendor/plugin"
