from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_managed_player_control_keeps_native_window_hidden() -> None:
    source = (
        ROOT / "cpp/infernux/platform/window/InxView.cpp"
    ).read_text(encoding="utf-8")
    init_start = source.index("void InxView::SDLInit()")
    init_end = source.index("void InxView::CreateSurface", init_start)
    init_body = source[init_start:init_end]
    assert '_INFERNUX_PLAYER_CONTROL_FILE' in init_body
    assert "ResolveWindowPresentationPolicy(hasControlChannel" in init_body
    assert "m_activateWhenShown = presentation.activateWhenShown" in init_body
    assert "if (!presentation.focusable)" in init_body
    assert "windowFlags |= SDL_WINDOW_NOT_FOCUSABLE" in init_body

    show_start = source.index("void InxView::Show()")
    show_end = source.index("bool InxView::PumpStartupEvents()", show_start)
    show_body = source[show_start:show_end]
    assert "ShowNativeWindow()" in show_body
    assert "SDL_HINT_WINDOW_ACTIVATE_WHEN_SHOWN" in show_body
