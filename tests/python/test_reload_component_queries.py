"""Public component queries bind consumers to the returned script's source."""

import importlib

import pytest

from infernux.components.missing_script import MissingScript
from infernux.runtime_services import get_runtime_service
from test_reload_coordinator import reload_project as reload_project


@pytest.mark.parametrize("query", ("name", "python_type", "all", "typed_all", "children", "parent"))
def test_component_query_retires_cached_consumer_with_failed_provider(reload_project, query):
    scene, coordinator, write = reload_project
    expressions = {
        "name": 'self.game_object.get_component("QueryProvider")',
        "python_type": 'self.game_object.get_py_component(type(self.game_object.get_py_components()[0]))',
        "all": 'next(item for item in self.game_object.get_components() if type(item).__name__ == "QueryProvider")',
        "typed_all": 'self.game_object.get_components("QueryProvider")[0]',
        "children": 'self.game_object.get_component_in_children("QueryProvider")',
        "parent": 'self.game_object.get_component_in_parent("QueryProvider")',
    }
    provider_source = '''
        import infernux as inx
        class QueryProvider(inx.InxComponent):
            def value(self):
                return 3
    '''
    provider_path = write("QueryProvider", provider_source)
    consumer_path = write("QueryConsumer", f'''
        import infernux as inx
        from infernux.runtime_services import install_runtime_service
        class QueryConsumer(inx.InxComponent):
            def start(self):
                self.peer = {expressions[query]}
                assert self.peer.value() in (3, 8)
                install_runtime_service("component-query-consumer", self)
    ''')
    coordinator.ensure_ready((provider_path, consumer_path))
    consumer_object = scene.create_game_object("查询组件使用者")
    provider_object = consumer_object
    if query in ("children", "parent"):
        provider_object = scene.create_game_object("查询组件提供者")
        child, parent = ((provider_object, consumer_object) if query == "children"
                         else (consumer_object, provider_object))
        child.set_parent(parent)
    provider = provider_object.add_component(importlib.import_module("QueryProvider").QueryProvider)
    consumer = consumer_object.add_component(importlib.import_module("QueryConsumer").QueryConsumer)
    assert provider._call_awake() and consumer._call_awake()
    assert provider._call_start() and consumer._call_start()
    assert get_runtime_service("component-query-consumer") is consumer
    assert consumer.peer is provider
    revision = coordinator.graph.revision
    with consumer._reload_owner.enter():
        for _ in range(32):
            consumer_object.get_component("QueryConsumer")
            provider_object.get_component("QueryProvider")
    assert coordinator.graph.revision == revision

    write("QueryProvider", "invalid (\n")
    coordinator.request((provider_path,))
    coordinator.compile_pending()
    coordinator.drain()
    assert consumer._reload_owner.closed and provider._reload_owner.closed
    assert get_runtime_service("component-query-consumer") is None
    assert coordinator.states[provider_path].status == coordinator.states[consumer_path].status == "failed"
    assert all(isinstance(item, MissingScript) for item in consumer_object.get_py_components())
    write("QueryProvider", provider_source.replace("return 3", "return 8"))
    coordinator.ensure_ready((provider_path,))
    current = get_runtime_service("component-query-consumer")
    assert current is not None and current is not consumer
    assert current.peer is not provider and current.peer.value() == 8
