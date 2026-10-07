"""One successful script revision publishes its entire component catalog."""
import sys

import pytest

from infernux.components.registry import (
    component_types_for_script_path, get_type_by_identity, unregister_component_script,
)
from infernux.components.script_loader import load_all_components_from_file


def write_components(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from infernux.components import InxComponent\nevents = []\n" + "".join(
            f"class {name}(InxComponent):\n"
            "    value: int = 7\n"
            "    def _infernux_startup_warmup(self):\n"
            f"        events.append('{name}')\n"
            for name in names
        ), encoding="utf-8",
    )


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "Assets" / "whole_file.py"
    yield path
    types = component_types_for_script_path(str(path))
    unregister_component_script(str(path))
    for cls in types:
        sys.modules.pop(cls.__module__, None)


@pytest.mark.parametrize("names", [("First",), ("First", "Second"), ("First", "Second", "Third")])
def test_all_declared_types_have_exact_published_identity(script, names):
    write_components(script, names)
    loaded = load_all_components_from_file(str(script), source_only=True)
    assert [cls.__name__ for cls in loaded] == list(names)
    assert component_types_for_script_path(str(script)) == tuple(loaded)
    for cls in loaded:
        assert get_type_by_identity(cls.__name__, cls._get_intrinsic_script_guid(), cls._get_type_guid()) is cls


@pytest.mark.parametrize("names", [(), ("First",), ("Second", "Third")])
def test_next_revision_retires_deleted_classes_and_replaces_survivors(script, names):
    write_components(script, ("First", "Second"))
    previous = load_all_components_from_file(str(script), source_only=True)
    write_components(script, names)
    loaded = load_all_components_from_file(str(script), source_only=True)
    assert component_types_for_script_path(str(script)) == tuple(loaded)
    assert [cls.__name__ for cls in loaded] == list(names)
    by_name = {cls.__name__: cls for cls in loaded}
    for cls in previous:
        assert get_type_by_identity(cls.__name__, cls._get_intrinsic_script_guid(), cls._get_type_guid()) is by_name.get(cls.__name__)


@pytest.mark.parametrize("entry", ["script", "project"])
@pytest.mark.parametrize("preloaded", [False, True])
def test_warmup_visits_every_file_type_after_initial_discovery(script, entry, preloaded):
    from infernux.engine.startup_warmup import (
        invalidate_source, run_project_script_warmups, run_script_warmup,
    )

    names = ("First", "Second", "Third")
    write_components(script, names)
    if preloaded:
        load_all_components_from_file(str(script), source_only=True)
    root = str(script.parent.parent)

    def run():
        if entry == "script":
            return run_script_warmup(str(script), project_path=root, scope="test-whole-file")
        return run_project_script_warmups(project_path=root, scope="test-whole-file")

    assert run() == 3
    registered = component_types_for_script_path(str(script))
    assert len(registered) == 3
    module = sys.modules[registered[0].__module__]
    assert module.events == list(names)
    assert run() == 0
    assert module.events == list(names)
    assert invalidate_source(str(script), project_path=root) == 3
    assert run() == 3
    assert module.events == list(names) * 2
