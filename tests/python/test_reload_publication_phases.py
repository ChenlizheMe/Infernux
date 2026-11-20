"""Publication barriers using real native bindings and ordinary preload APIs."""

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from test_reload_coordinator import reload_project as reload_project
from infernux.components.missing_script import MissingScript
from infernux.engine.path_utils import path_key
from infernux.runtime_services import get_runtime_service, remove_runtime_service


def _attach(scene, *names):
    obj = scene.create_game_object("Current publication")
    for name in names:
        component = getattr(importlib.import_module(name), name)()
        obj.add_py_component(component)
    for component in obj.get_py_components():
        assert component._call_awake()
    for component in obj.get_py_components():
        assert component._call_start()
    return obj


@pytest.mark.parametrize("change", ("shared_helper", "both_components"))
def test_all_sccs_bind_and_awake_before_any_start(reload_project, change):
    scene, coordinator, write = reload_project
    helper = write("PublicationHelper", "events = []\nversion = 1\n")
    first_source = '''
        import infernux as inx
        from PublicationHelper import events, version
        from infernux.components.missing_script import MissingScript
        class PublicationA(inx.InxComponent):
            def awake(self):
                events.append("A.awake")
            def start(self):
                other = self.game_object.get_component("PublicationB")
                assert other is not None and not isinstance(other, MissingScript)
                assert other.ready, "Every current Awake must precede Start"
                events.append("A.start")
    '''
    second_source = '''
        import infernux as inx
        from PublicationHelper import events, version
        class PublicationB(inx.InxComponent):
            def awake(self):
                self.ready = True
                events.append("B.awake")
            def start(self):
                events.append("B.start")
    '''
    first = write("PublicationA", first_source)
    second = write("PublicationB", second_source)
    coordinator.ensure_ready((helper, first, second))
    obj = _attach(scene, "PublicationA", "PublicationB")
    original = tuple(obj.get_py_components())
    previous_helper = sys.modules["PublicationHelper"]
    previous_helper.events.clear()
    if change == "shared_helper":
        write("PublicationHelper", "events = []\nversion = 2\n")
        coordinator.ensure_ready((helper,))
    else:
        write("PublicationA", textwrap.dedent(first_source) + "\nrevision = 2\n")
        write("PublicationB", textwrap.dedent(second_source) + "\nrevision = 2\n")
        coordinator.ensure_ready((first, second))
        assert sys.modules["PublicationHelper"] is previous_helper
    current = tuple(obj.get_py_components())
    assert all(now is not old and not isinstance(now, MissingScript) for now, old in zip(current, original))
    events = sys.modules["PublicationHelper"].events
    assert len(events) == 4
    assert set(events[:2]) == {"A.awake", "B.awake"}
    assert set(events[2:]) == {"A.start", "B.start"}
    assert all(coordinator.states[path].status == "active" for path in (first, second))


def _add_preload_adapter(coordinator):
    from infernux.plugins.preload import PreloadManager, _ReloadPreloads

    manager = PreloadManager(coordinator.graph.project_root)
    adapter = _ReloadPreloads(manager)
    coordinator.adapter.add(adapter)
    return manager, adapter


def test_preload_provider_is_visible_to_current_preloads_and_components(reload_project):
    scene, coordinator, write = reload_project
    provider = write("PublicationProvider", '''
        from infernux.lifecycle import InxPreload
        from infernux.runtime_services import install_runtime_service
        class Service:
            value = 7
        class Provider(InxPreload):
            def preload(self, context):
                install_runtime_service("publication-provider", Service())
    ''')
    consumer = write("PublicationConsumer", '''
        from infernux.lifecycle import InxPreload
        from infernux.runtime_services import get_runtime_service
        from PublicationProvider import Service
        class Consumer(InxPreload):
            def preload(self, context):
                self.service = get_runtime_service("publication-provider")
                assert isinstance(self.service, Service)
    ''')
    component = write("PublicationComponent", '''
        import infernux as inx
        from infernux.runtime_services import get_runtime_service
        from PublicationProvider import Service
        class PublicationComponent(inx.InxComponent):
            def awake(self):
                self.service = get_runtime_service("publication-provider")
                assert isinstance(self.service, Service)
    ''')
    manager, adapter = _add_preload_adapter(coordinator)
    try:
        coordinator.ensure_ready((provider, consumer, component))
        obj = _attach(scene, "PublicationComponent")
        original_service = obj.get_py_components()[0].service
        write("PublicationProvider", Path(provider).read_text(encoding="utf-8").replace("value = 7", "value = 9"))
        coordinator.ensure_ready((provider,))
        current = obj.get_py_components()[0]
        assert current.service.value == 9 and current.service is not original_service
        preload = next(state.instance for state in manager.states.values()
                       if path_key(state.source_path) == path_key(consumer))
        assert preload.service is current.service
        assert all(coordinator.states[path].status == "active" for path in (provider, consumer, component))
    finally:
        coordinator.adapter.remove(adapter)
        remove_runtime_service("publication-provider")


def test_source_input_import_during_preload_joins_this_publication_once(reload_project):
    _scene, coordinator, write = reload_project
    library = write("PublicationInput", "executions = 1\nvalue = 4\n")
    coordinator._source_inputs.add(library)
    preload = write("PublicationDynamicPreload", '''
        import importlib
        from infernux.lifecycle import InxPreload
        class Dynamic(InxPreload):
            def preload(self, context):
                name = "PublicationInput"
                self.library = importlib.import_module(name)
    ''')
    manager, adapter = _add_preload_adapter(coordinator)
    try:
        coordinator.ensure_ready((preload, library))
        current = next(iter(manager.states.values())).instance
        assert current.library is sys.modules["PublicationInput"]
        assert current.library.value == 4
        assert library not in coordinator._source_inputs
        edges = coordinator.graph.dependencies_of(preload)
        assert any(edge.target is not None and edge.target.module_name == "PublicationInput" for edge in edges)
        write("PublicationInput", "value = 8\n")
        coordinator.ensure_ready((library,))
        restored = next(iter(manager.states.values())).instance
        assert restored is not current and restored.library.value == 8
    finally:
        coordinator.adapter.remove(adapter)


def test_failed_initialization_retires_actual_consumers_and_preserves_unrelated_group(reload_project):
    _scene, coordinator, write = reload_project
    provider = write("PublicationFailProvider", '''
        from infernux.lifecycle import InxPreload
        from infernux.runtime_services import install_runtime_service
        class Provider(InxPreload):
            def preload(self, context):
                install_runtime_service("publication-failed", self)
                raise RuntimeError("current provider failed")
    ''')
    consumer = write("PublicationFailConsumer", '''
        from PublicationFailProvider import Provider
        from infernux.lifecycle import InxPreload
        from infernux.runtime_services import get_runtime_service
        class Consumer(InxPreload):
            def preload(self, context):
                assert get_runtime_service("publication-failed") is not None
    ''')
    unrelated = write("PublicationIndependent", "value = 11\n")
    manager, adapter = _add_preload_adapter(coordinator)
    try:
        coordinator.request((provider, consumer, unrelated))
        coordinator.compile_pending()
        assert coordinator.drain() == 3
        assert coordinator.states[provider].status == "failed"
        assert coordinator.states[consumer].status == "failed"
        assert coordinator.states[unrelated].status == "active"
        assert sys.modules["PublicationIndependent"].value == 11
        assert get_runtime_service("publication-failed") is None
        assert not manager.states
        assert "PublicationFailProvider" not in sys.modules
        assert "PublicationFailConsumer" not in sys.modules
        assert coordinator.drain() == 0
    finally:
        coordinator.adapter.remove(adapter)
        remove_runtime_service("publication-failed")


def test_accepted_helper_change_supersedes_queued_initialization_failure(reload_project):
    scene, coordinator, write = reload_project
    helper = write("PublicationQueueHelper", "value = 1\n")
    component = write("PublicationQueueComponent", '''
        import infernux as inx
        from PublicationQueueHelper import value
        class PublicationQueueComponent(inx.InxComponent): pass
    ''')
    coordinator.ensure_ready((component,))
    obj = _attach(scene, "PublicationQueueComponent")
    original = obj.get_py_components()[0]
    coordinator.queue_initialization_failure(component, RuntimeError("old initialization"))
    assert not original._reload_owner.active
    assert coordinator.pending_count > 0
    write("PublicationQueueHelper", "value = 2\n")
    coordinator.ensure_ready((helper,))
    current = obj.get_py_components()[0]
    assert current is not original and not isinstance(current, MissingScript)
    assert current.enabled and current._reload_owner.active
    assert coordinator.pending_count == 0 and coordinator.drain() == 0


def test_repeated_failure_preserves_authored_missing_component_enabled_state(reload_project):
    scene, coordinator, write = reload_project
    component = write("PublicationMissingEnabled", '''
        import infernux as inx
        class PublicationMissingEnabled(inx.InxComponent): pass
    ''')
    coordinator.ensure_ready((component,))
    obj = _attach(scene, "PublicationMissingEnabled")
    original = obj.get_py_components()[0]
    write("PublicationMissingEnabled", "invalid (\n")
    with pytest.raises(RuntimeError):
        coordinator.ensure_ready((component,))
    missing = obj.get_py_components()[0]
    assert isinstance(missing, MissingScript)
    # A separately restored scene may hold a MissingScript with authored data,
    # without the transient in-memory snapshot from this reload retirement.
    from infernux.components.missing_script import create_missing_script_component
    state = missing._reload_state
    restored = create_missing_script_component(
        type_name=state["type_name"], script_guid=state["script_guid"],
        type_guid=state["type_guid"], module_name=state["module_name"],
        qualified_name=state["qualified_name"], fields=state["fields"], error="unavailable",
    )
    restored._script_path = component
    restored._component_id = state["component_id"]
    restored._enabled = True
    assert obj.replace_py_component(missing, restored) is restored
    restored.enabled = True
    assert restored.enabled
    coordinator.adapter.fail((component,), "another current failure")
    coordinator.adapter.fail((component,), "the same unavailable source")
    write("PublicationMissingEnabled", '''
        import infernux as inx
        class PublicationMissingEnabled(inx.InxComponent): pass
    ''')
    coordinator.ensure_ready((component,))
    current = obj.get_py_components()[0]
    assert current is not original and not isinstance(current, MissingScript)
    assert current.enabled


@pytest.mark.parametrize("failure", ("compile", "exec"))
def test_dynamic_pending_dependency_failure_is_not_retried_and_repair_recovers_consumer(reload_project, failure):
    _scene, coordinator, write = reload_project
    provider = write("PublicationZDynamic", "invalid (\n" if failure == "compile" else '''
        from pathlib import Path
        receipt = Path(__file__).with_suffix(".executions")
        receipt.write_text(receipt.read_text() + "failed\\n" if receipt.exists() else "failed\\n")
        raise RuntimeError("current dynamic provider failed")
    ''')
    consumer = write("PublicationADynamic", '''
        import importlib
        name = "Publication" + "ZDynamic"
        linked = importlib.import_module(name)
    ''')
    healthy = write("PublicationStillIndependent", "value = 3\n")
    coordinator.request((consumer, provider, healthy))
    coordinator.compile_pending()
    coordinator.drain()
    assert coordinator.states[consumer].status == "failed"
    assert coordinator.states[provider].status == "failed"
    assert coordinator.states[healthy].status == "active"
    assert consumer in {coordinator.graph.module_for_path(item.path_key).source_path
                        for item in coordinator.graph.affected_closure((provider,))}
    if failure == "exec":
        assert Path(provider).with_suffix(".executions").read_text() == "failed\n"
    write("PublicationZDynamic", "value = 19\n")
    coordinator.ensure_ready((provider,))
    assert coordinator.states[consumer].status == "active"
    assert sys.modules["PublicationADynamic"].linked.value == 19


def test_dynamic_pending_import_uses_accepted_snapshot_instead_of_later_disk_bytes(reload_project):
    _scene, coordinator, write = reload_project
    provider = write("PublicationSnapshotProvider", "value = 17\n")
    consumer = write("PublicationSnapshotConsumer", '''
        import importlib
        name = "PublicationSnapshot" + "Provider"
        linked = importlib.import_module(name)
    ''')
    coordinator.request((consumer, provider))
    coordinator.compile_pending()
    write("PublicationSnapshotProvider", "raise RuntimeError('later unaccepted bytes')\n")
    coordinator.drain()
    assert sys.modules["PublicationSnapshotConsumer"].linked.value == 17
    assert coordinator.states[provider].status == "active"
    with pytest.raises(RuntimeError, match="not ready"):
        coordinator.ensure_ready((provider,))
    assert coordinator.states[consumer].status == "failed"


def test_failed_fresh_dynamic_chain_is_recorded_before_import_returns(reload_project):
    _scene, coordinator, write = reload_project
    leaf = write("PublicationFreshLeaf", "invalid (\n")
    middle = write("PublicationFreshMiddle", '''
        import importlib
        target = "PublicationFresh" + "Leaf"
        leaf = importlib.import_module(target)
    ''')
    root = write("PublicationFreshRoot", '''
        import importlib
        target = "PublicationFresh" + "Middle"
        middle = importlib.import_module(target)
    ''')
    # Only the root is requested. Both nested dependencies are first discovered
    # by real imports; the failed leaf never returns a module to the caller.
    with pytest.raises(RuntimeError, match="not ready"):
        coordinator.ensure_ready((root,))
    affected = {coordinator.graph.module_for_path(item.path_key).source_path
                for item in coordinator.graph.affected_closure((leaf,))}
    assert {root, middle, leaf}.issubset(affected)
    write("PublicationFreshLeaf", "value = 23\n")
    coordinator.ensure_ready((leaf,))
    assert sys.modules["PublicationFreshRoot"].middle.leaf.value == 23
    assert all(coordinator.states[path].status == "active" for path in (root, middle, leaf))
