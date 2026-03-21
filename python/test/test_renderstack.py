"""Tests for InfEngine.renderstack — InjectionPoint, RenderPass, ResourceBus, FullScreenEffect (real C++ backend)."""

from __future__ import annotations

import pytest

from InfEngine.renderstack.injection_point import InjectionPoint
from InfEngine.renderstack.resource_bus import ResourceBus
from InfEngine.renderstack.render_pass import RenderPass
from InfEngine.renderstack.fullscreen_effect import FullScreenEffect


# ══════════════════════════════════════════════════════════════════════
# InjectionPoint
# ══════════════════════════════════════════════════════════════════════

class TestInjectionPoint:
    def test_auto_display_name(self):
        ip = InjectionPoint(name="after_opaque")
        assert ip.display_name == "After Opaque"

    def test_explicit_display_name(self):
        ip = InjectionPoint(name="x", display_name="Custom")
        assert ip.display_name == "Custom"

    def test_default_resource_state(self):
        ip = InjectionPoint(name="test")
        assert ip.resource_state == {"color", "depth"}

    def test_custom_resource_state(self):
        ip = InjectionPoint(name="test", resource_state={"color"})
        assert ip.resource_state == {"color"}


# ══════════════════════════════════════════════════════════════════════
# ResourceBus
# ══════════════════════════════════════════════════════════════════════

class TestResourceBus:
    def test_empty_bus(self):
        bus = ResourceBus()
        assert bus.available_resources == set()
        assert bus.get("color") is None
        assert not bus.has("color")

    def test_set_and_get(self):
        bus = ResourceBus()
        bus.set("color", "handle_c")
        bus.set("depth", "handle_d")
        assert bus.get("color") == "handle_c"
        assert bus.has("depth")
        assert bus.available_resources == {"color", "depth"}

    def test_initial_resources(self):
        bus = ResourceBus(initial={"a": 1, "b": 2})
        assert bus.has("a")
        assert bus.get("b") == 2

    def test_snapshot(self):
        bus = ResourceBus(initial={"x": 10})
        snap = bus.snapshot()
        assert snap == {"x": 10}
        snap["x"] = 999
        assert bus.get("x") == 10

    def test_repr(self):
        bus = ResourceBus(initial={"a": 1, "b": 2})
        r = repr(bus)
        assert "a" in r
        assert "b" in r


# ══════════════════════════════════════════════════════════════════════
# RenderPass (base class)
# ══════════════════════════════════════════════════════════════════════

class TestRenderPass:
    def test_requires_name(self):
        with pytest.raises(ValueError, match="name"):
            RenderPass()

    def test_requires_injection_point(self):
        class NoIP(RenderPass):
            name = "test"
        with pytest.raises(ValueError, match="injection_point"):
            NoIP()

    def test_valid_subclass_constructs(self):
        class Good(RenderPass):
            name = "good"
            injection_point = "after_opaque"
        p = Good()
        assert p.enabled is True

    def test_validate_missing_resources(self):
        class NeedsColor(RenderPass):
            name = "need"
            injection_point = "x"
            requires = {"color", "depth"}
        p = NeedsColor()
        errors = p.validate({"color"})
        assert len(errors) > 0

    def test_validate_all_satisfied(self):
        class NeedsColor(RenderPass):
            name = "need"
            injection_point = "x"
            requires = {"color"}
        p = NeedsColor()
        errors = p.validate({"color", "depth"})
        assert errors == []


# ══════════════════════════════════════════════════════════════════════
# FullScreenEffect (base class)
# ══════════════════════════════════════════════════════════════════════

class TestFullScreenEffect:
    def test_default_resource_declarations(self):
        class TestFX(FullScreenEffect):
            name = "test_fx"
            injection_point = "before_post_process"
        fx = TestFX()
        assert "color" in fx.requires
        assert "color" in fx.modifies

    def test_enabled_default(self):
        class TestFX(FullScreenEffect):
            name = "test_fx"
            injection_point = "before_post_process"
        fx = TestFX()
        assert fx.enabled is True

    def test_disabled_construction(self):
        class TestFX(FullScreenEffect):
            name = "test_fx"
            injection_point = "before_post_process"
        fx = TestFX(enabled=False)
        assert fx.enabled is False
