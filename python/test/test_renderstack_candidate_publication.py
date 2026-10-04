"""Rejected pipeline generations retain their graph without becoming a retry loop."""
from datetime import datetime
import inspect
from types import SimpleNamespace

import pytest

from Infernux.core.assets import AssetManager
from Infernux.debug import DebugConsole, LogEntry, LogType
from Infernux.lib import ConsolePanel
from Infernux.renderstack.render_effect_compiler import RenderEffectArtifactRegistry
from Infernux.renderstack.render_stack import RenderStack


class PublicationContext:
    def __init__(self, samples=0):
        self.output_samples = samples
        self.applied = []
        self.reject = None

    def setup_camera_properties(self, camera):
        pass

    def cull(self, camera):
        return object()

    def apply_graph(self, graph):
        self.applied.append(graph)
        if graph is self.reject:
            raise ValueError("native graph rejected")

    def submit_culling(self, culling):
        pass

    def render_compiled(self, camera, revision):
        return True


@pytest.fixture
def publication(monkeypatch):
    generation = [10]
    monkeypatch.setattr(AssetManager, "refresh_pending", staticmethod(lambda: False))
    monkeypatch.setattr(RenderEffectArtifactRegistry, "topology_generation", classmethod(lambda cls: generation[0]))
    previous_console = DebugConsole._instance
    console = DebugConsole()
    panel = ConsolePanel()
    console.set_native_console(panel)
    stack = RenderStack()
    stack.pipeline
    monkeypatch.setattr(stack, "_collect_effect_parameter_updates", lambda context: (False, []))
    try:
        yield stack, generation, console, panel
    finally:
        console.set_native_console(None)
        console.clear()
        DebugConsole._instance = previous_console


def retain_graph(stack, samples=0):
    stack._select_graph_state(samples)
    state = stack._graph_state
    state.last_valid_description = SimpleNamespace(source_revision=1)
    state.artifact_generation = 9
    return state


def test_rejected_effect_generation_waits_for_an_actual_change(publication, monkeypatch):
    stack, generation, console, panel = publication
    state = retain_graph(stack)
    calls = []

    def rejected(**kwargs):
        calls.append(kwargs)
        raise ValueError("incompatible route effect policies for stages [route_probe]")

    monkeypatch.setattr(stack, "build_graph", rejected)
    context = PublicationContext()
    for _ in range(6):
        stack.render(context, object())
    assert len(calls) == 1
    assert state.description is state.last_valid_description
    assert "route_probe" in " ".join(stack.effect_compile_errors)
    assert panel.get_error_count() == 1

    generation[0] += 1
    stack.render(context, object())
    stack.render(context, object())
    assert len(calls) == 2
    assert panel.get_error_count() == 1


def test_initial_rejection_recovers_when_the_effect_generation_changes(publication, monkeypatch):
    stack, generation, console, panel = publication
    calls = []
    candidate = SimpleNamespace(source_revision=2)

    def build(**kwargs):
        calls.append(kwargs)
        if generation[0] == 10:
            raise ValueError("initial effect contract rejected")
        stack._graph_state.artifact_generation = generation[0]
        return candidate

    monkeypatch.setattr(stack, "build_graph", build)
    context = PublicationContext()
    with pytest.raises(RuntimeError, match="could not build"):
        stack.render(context, object())
    stack.render(context, object())
    assert len(calls) == 1
    assert stack._graph_state.last_valid_description is None
    generation[0] += 1
    stack.render(context, object())
    assert len(calls) == 2
    assert stack._graph_state.last_valid_description is candidate
    assert stack.effect_compile_errors == ()


def test_runtime_failure_is_deduplicated_and_clears_after_all_views_publish(publication, monkeypatch):
    stack, generation, console, panel = publication
    source = inspect.getsourcefile(type(stack.pipeline))
    console.log(LogEntry("authored runtime error", LogType.ERROR, datetime.now(), source_file=source))
    valid_probe = stack._build_full_topology_probe()
    retain_graph(stack, 0)
    retain_graph(stack, 4)

    def rejected(**kwargs):
        raise ValueError("duplicate stage declaration")

    monkeypatch.setattr(stack, "build_graph", rejected)
    monkeypatch.setattr(stack.pipeline, "define_topology", lambda graph: (_ for _ in ()).throw(ValueError("duplicate stage declaration")))
    stack.invalidate_graph()
    assert stack._build_full_topology_probe() is valid_probe
    for samples in (0, 4):
        stack.render(PublicationContext(samples), object())
    diagnostics = [entry for entry in console.get_entries() if "duplicate stage declaration" in entry.message]
    assert len(diagnostics) == 1
    assert diagnostics[0].source_file == source
    assert diagnostics[0].stack_trace == ""
    assert panel.get_error_count() == 2

    monkeypatch.undo()
    # A successful Inspector probe cannot certify publication of the runtime graph.
    stack._build_full_topology_probe()
    assert panel.get_error_count() == 2
    monkeypatch.setattr(stack, "_collect_effect_parameter_updates", lambda context: (False, []))
    monkeypatch.setattr(AssetManager, "refresh_pending", staticmethod(lambda: False))
    monkeypatch.setattr(RenderEffectArtifactRegistry, "topology_generation", classmethod(lambda cls: generation[0]))

    def valid(**kwargs):
        stack._graph_state.artifact_generation = generation[0]
        return SimpleNamespace(source_revision=2)

    monkeypatch.setattr(stack, "build_graph", valid)
    stack.invalidate_graph()
    stack.render(PublicationContext(0), object())
    assert panel.get_error_count() == 2
    stack.render(PublicationContext(4), object())
    assert stack.effect_compile_errors == ()
    assert [entry.message for entry in console.get_entries()] == ["authored runtime error"]
    assert [entry["message"] for entry in panel._get_visible_log_snapshot(10)] == ["authored runtime error"]


def test_valid_probe_does_not_hide_native_publication_rejection(publication, monkeypatch):
    stack, generation, console, panel = publication
    state = retain_graph(stack)
    candidate = SimpleNamespace(source_revision=2)

    def build(**kwargs):
        stack._graph_state.artifact_generation = generation[0]
        return candidate

    monkeypatch.setattr(stack, "build_graph", build)
    context = PublicationContext()
    context.reject = candidate
    stack.render(context, object())
    assert state.description is state.last_valid_description
    stack._build_full_topology_probe()
    assert "native graph rejected" in " ".join(stack.effect_compile_errors)
    assert panel.get_error_count() == 1
    for _ in range(3):
        stack.render(context, object())
    assert len(context.applied) == 2
    context.reject = None
    stack.invalidate_graph()
    stack.render(context, object())
    assert state.last_valid_description is candidate
    assert stack.effect_compile_errors == ()
    assert panel.get_error_count() == 0
