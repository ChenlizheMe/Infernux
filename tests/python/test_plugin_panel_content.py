from types import SimpleNamespace

import pytest

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


def test_document_images_use_source_illustration_renderer(monkeypatch):
    panel = PluginPanel()
    requests = []
    monkeypatch.setattr(panel_module, "_metric", lambda ctx, value: value)
    monkeypatch.setattr(panel_module, "render_document_image", lambda *args: requests.append(args) or True)
    ctx = SimpleNamespace(get_content_region_avail_width=lambda: 900)
    manager = SimpleNamespace(content_asset_path=lambda *args: "/downloaded/plugin_pages/overview.png")
    panel._render_markdown_image(ctx, manager, {}, {}, {"source": "overview.png"})
    assert requests == [(ctx, panel, "/downloaded/plugin_pages/overview.png", 720, 360)]


def test_document_image_preserves_aspect_and_ui_preview_ownership(monkeypatch, tmp_path):
    from infernux.engine.ui import asset_resource_preview as preview

    path = tmp_path / "diagram.png"
    path.write_bytes(b"source illustration")
    calls = []
    draws = []
    native = SimpleNamespace(
        query_or_schedule_texture_preview=lambda *args, **kwargs:
            calls.append((args, kwargs)) or (7, 1280, 520),
    )
    monkeypatch.setattr(preview, "_resolve_native_engine", lambda panel: native)
    ctx = SimpleNamespace(image=lambda *args: draws.append(args))

    assert preview.render_document_image(ctx, None, str(path), 720, 360)
    args, settings = calls[0]
    assert args[0] == f"document|{path}"
    assert args[2] == path.stat().st_mtime_ns
    assert settings["authoring"] is False
    assert settings["srgb"] is True
    assert settings["texture_type"] == "ui"
    assert settings["max_size"] >= 1280
    assert draws == [(7, 720, 292.5)]
    assert args[0] not in preview._AUTHORING_PREVIEW_KEYS


def test_document_image_waits_for_async_upload(monkeypatch, tmp_path):
    from infernux.engine.ui import asset_resource_preview as preview

    path = tmp_path / "pending.png"
    path.write_bytes(b"source illustration")
    native = SimpleNamespace(query_or_schedule_texture_preview=lambda *args, **kwargs: (0, 0, 0))
    monkeypatch.setattr(preview, "_resolve_native_engine", lambda panel: native)
    ctx = SimpleNamespace(image=lambda *args: pytest.fail("unpublished texture was drawn"))
    assert not preview.render_document_image(ctx, None, str(path), 720, 360)


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
