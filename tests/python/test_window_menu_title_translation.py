"""Cached native menu descriptors retain translation keys across locale changes."""
from infernux.lib import MenuBarPanel, WindowTypeInfo


def test_cached_window_title_resolves_the_current_locale():
    descriptor = WindowTypeInfo()
    descriptor.type_id = "my_studio.hello_plugin.panel"
    descriptor.display_name = "Hello Plugin"
    descriptor.title_key = "my_studio.hello_plugin.panel_title"
    locales = {"zh": {descriptor.title_key: "你好插件"}, "en": {descriptor.title_key: "Hello Plugin"}}
    current = "zh"
    menu = MenuBarPanel()
    menu.translate = lambda key: locales[current][key]
    menu.has_translation = lambda key: key in locales[current]
    assert menu.resolve_menu_label(descriptor.title_key, descriptor.display_name) == "你好插件"
    current = "en"
    assert menu.resolve_menu_label(descriptor.title_key, descriptor.display_name) == "Hello Plugin"
    current = "zh"
    assert menu.resolve_menu_label(descriptor.title_key, descriptor.display_name) == "你好插件"
    assert descriptor.type_id == "my_studio.hello_plugin.panel"


def test_window_title_without_a_translation_key_keeps_its_authored_label():
    descriptor = WindowTypeInfo()
    descriptor.display_name = "Literal Tool"
    menu = MenuBarPanel()
    menu.has_translation = lambda key: False
    assert descriptor.title_key == ""
    assert menu.resolve_menu_label(descriptor.title_key, descriptor.display_name) == "Literal Tool"
