"""FORGE SIGNAL theme for Infernux Hub.

The Hub shares the website's visual language — Swiss grid and type,
cassette-futurist controls, NASA-punk plates — anchored on the Infernux red.
It uses a single typeface, Space Grotesk, in three weights; hierarchy comes
from size, weight, case and tracking, never from a second font. Painted
primitives live in ``view/forge.py``; dialogs in ``view/dialogs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


FONT_FAMILIES = ("Space Grotesk", "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC")
FONT_STACK = ", ".join(f'"{family}"' for family in FONT_FAMILIES) + ", sans-serif"

# Logical-pixel type scale; Qt scales it with the display's DPI.
SIZE_DISPLAY = 32
SIZE_TITLE = 22
SIZE_HEADING = 16
SIZE_BODY = 13
SIZE_SMALL = 12
SIZE_LABEL = 10


@dataclass(frozen=True)
class ThemePalette:
    bg_base: str
    bg_surface: str
    bg_surface_hover: str
    bg_surface_selected: str
    bg_input: str
    text_primary: str
    text_secondary: str
    text_muted: str
    border: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    danger: str
    sidebar_bg: str
    sidebar_border: str
    nav_hover: str
    nav_active: str
    border_hover: str
    button_surface: str
    button_hover: str
    button_pressed: str
    disabled_surface: str
    disabled_text: str
    accent_fill: str
    accent_deep: str
    bg_deep: str
    grid: str
    signal: str
    warn: str
    text_mono: str
    chip: str
    label_paper: str
    label_ink: str
    overlay: str


_DARK_PALETTE = ThemePalette(
    bg_base="#0e0f10",
    bg_surface="#151617",
    bg_surface_hover="#1b1d1e",
    bg_surface_selected="#211617",
    bg_input="#0a0b0c",
    text_primary="#f1ece3",
    text_secondary="#c2bfb8",
    text_muted="#8f8c86",
    border="#2a2c2e",
    accent="#ff6e6e",
    accent_hover="#d4515a",
    accent_pressed="#a9323c",
    accent_text="#fff6f0",
    danger="#ff7a72",
    sidebar_bg="#09090a",
    sidebar_border="#1f2122",
    nav_hover="#141516",
    nav_active="#1d1314",
    border_hover="#46494b",
    button_surface="#18191a",
    button_hover="#232526",
    button_pressed="#101112",
    disabled_surface="#121314",
    disabled_text="#5d5b57",
    accent_fill="#c8444b",
    accent_deep="#8f2632",
    bg_deep="#070808",
    grid="#f4eadb",
    signal="#83b8ad",
    warn="#e3b15f",
    text_mono="#dedbd2",
    chip="#1e2022",
    label_paper="#e9e1d2",
    label_ink="#17161a",
    overlay="#050505",
)

_LIGHT_PALETTE = ThemePalette(
    bg_base="#ebe5d9",
    bg_surface="#f6f2ea",
    bg_surface_hover="#f0eadf",
    bg_surface_selected="#f3e1db",
    bg_input="#fbf8f2",
    text_primary="#1b1a18",
    text_secondary="#46423c",
    text_muted="#6c665d",
    border="#cfc6b6",
    accent="#b8313b",
    accent_hover="#b53a43",
    accent_pressed="#8f2632",
    accent_text="#fff8f2",
    danger="#a92a34",
    sidebar_bg="#e1d9ca",
    sidebar_border="#cbc1af",
    nav_hover="#d9d0bf",
    nav_active="#f1d9d3",
    border_hover="#9f9583",
    button_surface="#f6f2ea",
    button_hover="#ece5d8",
    button_pressed="#ddd5c6",
    disabled_surface="#e4ddd0",
    disabled_text="#9a9387",
    accent_fill="#c8444b",
    accent_deep="#8f2632",
    bg_deep="#ded6c6",
    grid="#3a2f22",
    signal="#3c7a6d",
    warn="#9a6a16",
    text_mono="#2b2925",
    chip="#e4ddcf",
    label_paper="#1d1c1a",
    label_ink="#f3ede2",
    overlay="#1a1712",
)


class StyleManager:
    """Provides dynamic QSS stylesheets for light/dark modes."""

    @staticmethod
    def palette(is_dark: bool) -> ThemePalette:
        return _DARK_PALETTE if is_dark else _LIGHT_PALETTE

    @staticmethod
    def get_stylesheet(is_dark: bool) -> str:
        p = StyleManager.palette(is_dark)
        return f"""
            * {{
                font-family: {FONT_STACK};
                font-size: {SIZE_BODY}px;
                font-weight: 400;
                color: {p.text_primary};
                outline: none;
            }}
            QMainWindow, QWidget#central, QDialog {{
                background-color: {p.bg_base};
            }}
            QAbstractItemView, QTextEdit, QPlainTextEdit {{
                background-color: {p.bg_input};
                alternate-background-color: {p.bg_surface};
                color: {p.text_primary};
                selection-background-color: {p.accent_fill};
                selection-color: {p.accent_text};
                border: 1px solid {p.border};
            }}
            QHeaderView::section {{
                background-color: {p.bg_surface};
                color: {p.text_secondary};
                border: none;
                padding: 4px;
            }}
            QToolTip {{
                background-color: {p.label_paper};
                color: {p.label_ink};
                border: none;
                padding: 5px 9px;
                font-size: {SIZE_SMALL}px;
                font-weight: 500;
            }}

            /* ── Type ── */
            QLabel#pageKicker, QLabel#monoKicker, QLabel#discussionEyebrow {{
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 700;
                color: {p.accent};
            }}
            QLabel#pageTitle {{
                font-size: {SIZE_DISPLAY}px;
                font-weight: 700;
                color: {p.text_primary};
            }}
            QLabel#sectionTitle, QLabel#communityFeedTitle {{
                font-size: {SIZE_HEADING + 1}px;
                font-weight: 700;
                color: {p.text_primary};
            }}
            QLabel#pageSubtitle, QLabel#dialogSubtitle, QLabel#discussionDescription {{
                font-size: {SIZE_BODY}px;
                color: {p.text_secondary};
            }}
            QLabel#monoLabel, QLabel#monoMeta, QLabel#telemetryLine, QLabel#columnHeader,
            QLabel#communityTopicStats, QLabel#communityFeedStatus, QLabel#discussionAddress {{
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 500;
                color: {p.text_muted};
            }}
            QLabel#sidebarSection {{
                padding-left: 20px;
            }}
            QLabel#columnHeader, QLabel#sidebarSection {{
                font-size: {SIZE_LABEL}px;
                font-weight: 700;
                color: {p.text_muted};
            }}
            QLabel#monoValue, QLabel#queueSummary {{
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 700;
                color: {p.text_mono};
            }}
            QLabel#dialogTitle, QLabel#discussionHeading, QLabel#emptyTitle {{
                font-size: {SIZE_TITLE}px;
                font-weight: 700;
                color: {p.text_primary};
            }}
            QLabel#fieldLabel {{
                font-size: {SIZE_LABEL}px;
                font-weight: 700;
                color: {p.text_muted};
            }}
            QLabel#dialogErrorHint, QLabel#communityFeedStatus[kind="error"] {{
                color: {p.danger};
            }}
            QLabel#dialogBody {{
                font-size: {SIZE_BODY + 1}px;
                color: {p.text_secondary};
            }}

            /* ── Legacy message boxes (kept readable if any remain) ── */
            QMessageBox {{
                padding: 18px;
            }}
            QMessageBox QLabel {{
                font-size: 15px;
                min-width: 360px;
                padding: 6px 4px 10px 4px;
            }}
            QMessageBox QPushButton {{
                min-width: 88px;
                min-height: 34px;
                padding: 0 16px;
                font-size: 14px;
                font-weight: 600;
            }}

            /* ── Menus ── */
            QMenu {{
                background-color: {p.bg_surface};
                color: {p.text_primary};
                border: 1px solid {p.border_hover};
                padding: 5px 0;
            }}
            QMenu::item {{
                background: transparent;
                padding: 8px 28px 8px 16px;
                margin: 0 4px;
                font-weight: 500;
            }}
            QMenu::item:selected {{
                background-color: {p.nav_active};
                color: {p.text_primary};
                border-left: 2px solid {p.accent_fill};
                padding-left: 14px;
            }}
            QMenu::item:disabled {{
                color: {p.disabled_text};
            }}
            QMenu::separator {{
                height: 1px;
                background: {p.border};
                margin: 5px 10px;
            }}
            QMenu#dangerMenuItem::item {{ color: {p.danger}; }}

            /* ── Sidebar ── */
            QWidget#sidebar {{
                background-color: {p.sidebar_bg};
                border-right: 1px solid {p.sidebar_border};
            }}
            QWidget#sidebarHeader, QLabel#sidebarLogo {{
                background: transparent;
            }}
            QLabel#sidebarTitle {{
                font-size: 18px;
                font-weight: 700;
                color: {p.text_primary};
            }}
            QLabel#sidebarSubtitle {{
                font-size: {SIZE_LABEL}px;
                font-weight: 700;
                color: {p.text_muted};
            }}
            QPushButton#navItem {{
                background: transparent;
                border: none;
                text-align: left;
                padding: 0;
                font-size: 14px;
                color: {p.text_secondary};
            }}

            /* ── Scroll bars ── */
            QScrollBar:vertical {{
                background: transparent;
                width: 9px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {p.border};
                min-height: 32px;
                margin: 0 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {p.accent_fill};
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 9px;
            }}
            QScrollBar::handle:horizontal {{
                background: {p.border};
                min-width: 32px;
                margin: 2px 0;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
            QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollArea#projectScrollArea,
            QWidget#projectViewport,
            QWidget#projectListContainer,
            QScrollArea#installScrollArea,
            QWidget#installViewport,
            QWidget#installListContainer,
            QScrollArea#settingsScrollArea,
            QWidget#settingsViewport,
            QWidget#settingsContent {{
                background-color: {p.bg_base};
            }}

            /* ── Buttons: square plates, 1px rule, press sinks one pixel ── */
            QPushButton {{
                background-color: {p.button_surface};
                color: {p.text_primary};
                border: 1px solid {p.border_hover};
                border-radius: 0;
                padding: 0 16px;
                min-height: 32px;
                font-size: {SIZE_BODY}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {p.button_hover};
                border-color: {p.text_muted};
            }}
            QPushButton:pressed {{
                background-color: {p.button_pressed};
                padding-top: 2px;
            }}
            QPushButton:focus {{
                border-color: {p.accent};
            }}
            QPushButton:disabled {{
                background-color: {p.disabled_surface};
                color: {p.disabled_text};
                border-color: {p.border};
            }}
            QPushButton#primaryBtn, QPushButton#createBtn {{
                background-color: {p.accent_fill};
                color: {p.accent_text};
                border: 1px solid {p.accent_fill};
                font-weight: 700;
            }}
            QPushButton#primaryBtn:hover, QPushButton#createBtn:hover {{
                background-color: {p.accent_hover};
                border-color: {p.accent};
            }}
            QPushButton#primaryBtn:pressed, QPushButton#createBtn:pressed {{
                background-color: {p.accent_pressed};
            }}
            QPushButton#primaryBtn:disabled, QPushButton#createBtn:disabled {{
                background-color: {p.disabled_surface};
                color: {p.disabled_text};
                border-color: {p.border};
            }}
            QPushButton#dangerBtn {{
                color: {p.danger};
                border: 1px solid {p.accent_deep};
                background-color: transparent;
                font-weight: 500;
            }}
            QPushButton#dangerBtn:hover, QPushButton#dangerSolidBtn {{
                background-color: {p.accent_pressed};
                color: {p.accent_text};
                border: 1px solid {p.accent_pressed};
                font-weight: 700;
            }}
            QPushButton#dangerSolidBtn:hover {{
                background-color: {p.accent_fill};
                border-color: {p.accent};
            }}
            QPushButton#ghostBtn, QPushButton#iconBtn, QPushButton#cardOpenBtn {{
                background: transparent;
                border: 1px solid transparent;
                color: {p.text_secondary};
                padding: 0 10px;
                font-weight: 500;
            }}
            QPushButton#ghostBtn:hover, QPushButton#iconBtn:hover, QPushButton#cardOpenBtn:hover {{
                border-color: {p.border_hover};
                color: {p.text_primary};
                background: {p.button_hover};
            }}
            QPushButton#ghostBtn:disabled, QPushButton#cardOpenBtn:disabled {{
                background: transparent;
                border-color: transparent;
            }}
            QPushButton#launchBtn {{
                background-color: transparent;
                color: {p.accent};
                border: 1px solid {p.accent_deep};
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 700;
                padding: 0 12px;
                min-height: 28px;
            }}
            QPushButton#launchBtn:hover {{
                background-color: {p.accent_fill};
                color: {p.accent_text};
                border-color: {p.accent_fill};
            }}
            QPushButton#segmentBtn {{
                background: transparent;
                border: 1px solid {p.border};
                border-right: none;
                color: {p.text_secondary};
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 700;
                padding: 0 12px;
                min-height: 28px;
            }}
            QPushButton#segmentBtn[last="true"] {{
                border-right: 1px solid {p.border};
            }}
            QPushButton#segmentBtn:hover {{
                color: {p.text_primary};
                background: {p.button_hover};
            }}
            QPushButton#segmentBtn:checked {{
                background: {p.label_paper};
                color: {p.label_ink};
                border-color: {p.label_paper};
            }}
            QPushButton#updatePill {{
                background-color: {p.accent_fill};
                color: {p.accent_text};
                border: none;
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 700;
                text-align: left;
                padding: 0 12px;
                min-height: 30px;
            }}
            QPushButton#updatePill:hover {{
                background-color: {p.accent_hover};
            }}

            /* ── Inputs ── */
            QLineEdit {{
                background-color: {p.bg_input};
                color: {p.text_primary};
                border: 1px solid {p.border};
                border-radius: 0;
                padding: 7px 11px;
                font-size: {SIZE_BODY}px;
                selection-background-color: {p.accent_fill};
                selection-color: {p.accent_text};
            }}
            QLineEdit:hover {{
                border-color: {p.border_hover};
            }}
            QLineEdit:focus {{
                border-color: {p.accent};
                background-color: {p.bg_surface};
            }}
            QLineEdit:disabled, QLineEdit:read-only {{
                color: {p.text_secondary};
            }}
            QLineEdit[invalid="true"] {{
                border-color: {p.danger};
            }}
            QComboBox {{
                background-color: {p.bg_input};
                color: {p.text_primary};
                border: 1px solid {p.border};
                border-radius: 0;
                padding: 6px 36px 6px 11px;
                font-size: {SIZE_BODY}px;
                font-weight: 500;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: {p.border_hover};
            }}
            QComboBox:focus, QComboBox:on {{
                border-color: {p.accent};
            }}
            QComboBox:disabled {{
                color: {p.disabled_text};
                background-color: {p.disabled_surface};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border: none;
                border-left: 1px solid {p.border};
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0;
                height: 0;
            }}
            QComboBox QAbstractItemView {{
                background-color: {p.bg_surface};
                color: {p.text_primary};
                border: 1px solid {p.border_hover};
                selection-background-color: {p.nav_active};
                selection-color: {p.text_primary};
                padding: 3px 0;
                outline: 0;
            }}
            QCheckBox {{
                spacing: 9px;
                color: {p.text_secondary};
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 1px solid {p.border_hover};
                background: {p.bg_input};
            }}
            QCheckBox::indicator:hover {{
                border-color: {p.accent};
            }}
            QCheckBox::indicator:checked {{
                background: {p.accent_fill};
                border-color: {p.accent_fill};
            }}

            /* ── Cards and rows (painted surfaces; QSS only clears backgrounds) ── */
            QFrame#projectCard, QFrame#versionCard, QFrame#versionRow,
            QFrame#settingsCard, QFrame#discussionHero, QFrame#communityTopic {{
                background: transparent;
                border: none;
            }}
            QLabel#cardName {{
                font-size: 15px;
                font-weight: 700;
            }}
            QFrame#projectCard[unavailable="true"] QLabel#cardName {{
                color: {p.text_muted};
            }}
            QLabel#cardPath {{
                font-size: {SIZE_SMALL - 1}px;
                color: {p.text_muted};
            }}
            QLabel#cardDate, QLabel#cardVersion {{
                font-size: {SIZE_SMALL - 1}px;
                font-weight: 500;
                color: {p.text_muted};
            }}
            QLabel#engineCaption {{
                font-size: {SIZE_LABEL}px;
                font-weight: 500;
                color: {p.text_muted};
            }}
            QLabel#engineCaption[kind="warning"] {{ color: {p.warn}; }}
            QLabel#engineCaption[kind="error"] {{ color: {p.danger}; }}

            /* Tape-label chips: capitals on a printed strip. */
            QLabel#chip, QLabel#projectVersion, QLabel#projectStatus, QLabel#installedBadge, QLabel#versionBadge {{
                font-size: {SIZE_LABEL + 1}px;
                font-weight: 700;
                padding: 2px 7px;
                color: {p.text_secondary};
                background-color: {p.chip};
                border: 1px solid {p.border};
            }}
            QLabel#chip[kind="ready"], QLabel#projectVersion[kind="ready"] {{
                color: {p.text_mono};
            }}
            QLabel#chip[kind="ok"], QLabel#installedBadge {{
                color: {p.signal};
                border-color: {p.signal};
                background-color: transparent;
            }}
            QLabel#chip[kind="warning"], QLabel#projectVersion[kind="warning"], QLabel#installedBadge[kind="warning"] {{
                color: {p.warn};
                border-color: {p.warn};
                background-color: transparent;
            }}
            QLabel#chip[kind="error"], QLabel#projectStatus[kind="error"] {{
                color: {p.danger};
                border-color: {p.accent_deep};
                background-color: transparent;
            }}
            QLabel#chip[kind="active"], QLabel#projectVersion[kind="active"], QLabel#projectStatus[kind="active"] {{
                color: {p.accent_text};
                background-color: {p.accent_fill};
                border-color: {p.accent_fill};
            }}
            QLabel#chip[kind="paper"], QLabel#versionBadge {{
                color: {p.label_ink};
                background-color: {p.label_paper};
                border-color: {p.label_paper};
            }}
            QLabel#versionBadge {{
                font-size: 15px;
                padding: 4px 10px;
            }}

            /* ── Installs tabs ── */
            QTabWidget#installTabs::pane {{
                border: none;
                border-top: 1px solid {p.border};
                top: -1px;
            }}
            QTabWidget#installTabs QTabBar::tab {{
                background: transparent;
                color: {p.text_muted};
                font-size: {SIZE_SMALL}px;
                font-weight: 700;
                padding: 9px 16px 8px 16px;
                margin-right: 2px;
                border: 1px solid transparent;
                border-bottom: none;
            }}
            QTabWidget#installTabs QTabBar::tab:selected {{
                color: {p.label_ink};
                background: {p.label_paper};
            }}
            QTabWidget#installTabs QTabBar::tab:hover:!selected {{
                color: {p.text_primary};
                border-color: {p.border};
            }}

            /* ── Empty states / settings ── */
            QLabel#emptyHint {{
                font-size: {SIZE_BODY}px;
                color: {p.text_muted};
                padding: 4px;
            }}
            QFrame#emptyState {{
                background-color: transparent;
                border: 1px dashed {p.border_hover};
                min-height: 220px;
            }}
            QLabel#settingsLabel {{
                font-size: 14px;
                font-weight: 700;
            }}
            QLabel#settingsDescription {{
                font-size: {SIZE_SMALL}px;
                color: {p.text_secondary};
            }}
            QLabel#settingsPath {{
                font-size: {SIZE_SMALL - 1}px;
                color: {p.text_muted};
            }}
            QFrame#settingsRule, QFrame#telemetryRule, QFrame#dialogRule {{
                background-color: {p.border};
                border: none;
            }}

            /* ── Community ── */
            QLabel#communityTopicTitle {{
                font-size: 14px;
                font-weight: 500;
            }}
            QLabel#topicRank {{
                font-size: 18px;
                font-weight: 700;
                color: {p.accent};
            }}

            /* ── Transfer deck ── */
            QFrame#installQueuePanel, QFrame#installQueuePopup {{
                background-color: {p.bg_surface};
                border: 1px solid {p.border_hover};
            }}
            QFrame#installQueuePanel:hover {{
                border-color: {p.accent_fill};
            }}
            QFrame#queueRow {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {p.border};
            }}
            QLabel#queueTitle {{
                font-size: {SIZE_BODY}px;
                font-weight: 700;
            }}

            /* ── Dialog shell (view/dialogs.py) ── */
            QFrame#dialogDetail {{
                background-color: {p.bg_input};
                border: 1px solid {p.border};
            }}
            QPlainTextEdit#dialogDetailText {{
                border: none;
                background: transparent;
                font-size: {SIZE_SMALL - 1}px;
                color: {p.text_secondary};
            }}
            QFrame#subjectCard {{
                background-color: {p.bg_input};
                border: 1px solid {p.border};
                border-left: 3px solid {p.accent_fill};
            }}
            QLabel#toastText {{
                font-size: {SIZE_BODY}px;
                font-weight: 500;
                color: {p.text_primary};
            }}

            /* ── Progress ── */
            QProgressBar {{
                background-color: {p.button_surface};
                border: none;
                border-radius: 0;
                height: 6px;
            }}
            QProgressBar::chunk {{
                background-color: {p.accent_fill};
                border-radius: 0;
            }}
        """
