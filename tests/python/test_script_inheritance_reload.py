"""Hot reload preserves the published inheritance graph and native instances."""
import sys

import pytest

from infernux.components.fields import get_serialized_fields
from infernux.components.registry import (
    component_types_for_script_path, restore_component_registry_state, snapshot_component_registry_state,
)
from infernux.components.script_loader import (
    ComponentBodyReloadRequest, ScriptReloadRejected,
    load_all_components_from_file, stage_component_body_reload_batch,
)
from infernux.engine.project_context import get_project_root, set_project_root


@pytest.fixture
def project(tmp_path):
    prior_root = get_project_root()
    prior_registry = snapshot_component_registry_state()
    previous = {name: sys.modules.get(name) for name in ("reload_tree_base", "reload_tree_child")}
    assets = tmp_path / "Assets"
    assets.mkdir()
    set_project_root(str(tmp_path))
    try:
        yield assets
    finally:
        restore_component_registry_state(prior_registry)
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        set_project_root(prior_root)


def write_tree(assets, cross_file, version, add_field=False, changed_base=False):
    base = assets / "reload_tree_base.py"
    child = assets / "reload_tree_child.py"
    base_source = (
        "from infernux.components import InxComponent\n"
        "class Base(InxComponent):\n"
        "    value: int = 7\n"
        f"    def marker(self): return '{version}:Base'\n"
        + ("    extra: int = 9\n" if add_field else "")
    )
    child_source = (
        f"class Child({'InxComponent' if changed_base else 'Base'}):\n"
        f"    def marker(self): return super().marker() + ':{version}:Child'\n"
        "class Leaf(Child):\n"
        f"    def marker(self): return super().marker() + ':{version}:Leaf'\n"
        "class Unmounted(Base):\n"
        f"    def marker(self): return super().marker() + ':{version}:Unmounted'\n"
    )
    if version == "new":
        child_source += (
            "class Added(Base):\n"
            "    def marker(self): return super().marker() + ':new:Added'\n"
        )
    if cross_file:
        base.write_text(base_source, encoding="utf-8")
        child.write_text("from reload_tree_base import Base\nfrom infernux.components import InxComponent\n" + child_source, encoding="utf-8")
        return (base, child)
    base.write_text(base_source + child_source, encoding="utf-8")
    return (base,)


@pytest.mark.parametrize("cross_file,reverse", [(False,False), (True,False), (True,True)])
@pytest.mark.parametrize("mount_base", [False,True])
@pytest.mark.parametrize("add_field", [False,True])
def test_batch_reloads_inherited_methods_fields_and_super_without_changing_live_identity(
    scene, project, cross_file, reverse, mount_base, add_field,
):
    paths = write_tree(project, cross_file, "old")
    types = {cls.__name__:cls for path in paths for cls in load_all_components_from_file(str(path), source_only=True)}
    mounted = ("Base", "Child", "Leaf") if mount_base else ("Leaf",)
    instances = {name:scene.create_game_object(name).add_component(types[name]) for name in mounted}
    for instance in instances.values():
        instance.value = 31
    snapshots = {name:instance._serialize_fields_document() for name,instance in instances.items()}
    methods = {name:instance.marker() for name,instance in instances.items()}
    modules = {cls.__module__:sys.modules[cls.__module__] for cls in types.values()}
    write_tree(project, cross_file, "new", add_field=add_field)
    order = paths[::-1] if reverse else paths
    transaction = stage_component_body_reload_batch(tuple(
        ComponentBodyReloadRequest(str(path)) for path in order
    ))
    try:
        assert {name:instance.marker() for name,instance in instances.items()} == methods
        transaction.commit()
        for name,instance in instances.items():
            assert type(instance) is types[name]
            assert instance._component_id == snapshots[name]["__component_id__"]
            assert instance.value == 31
            assert "old" not in instance.marker()
            assert instance.marker().startswith("new:Base")
            assert instance.extra == 9 if add_field else "extra" not in get_serialized_fields(type(instance))
        assert types["Leaf"].__bases__ == (types["Child"],)
        assert types["Child"].__bases__ == (types["Base"],)
        published = {cls.__name__:cls for path in paths for cls in component_types_for_script_path(str(path))}
        for name in ("Unmounted", "Added"):
            cls = published[name]
            assert issubclass(cls, types["Base"])
            component = scene.create_game_object(name).add_component(cls)
            assert component.value == 7
            assert component.marker() == f"new:Base:new:{name}"
            if add_field:
                assert component.extra == 9
            component.game_object.remove_py_component(component)
    finally:
        transaction.rollback()
    for name,instance in instances.items():
        assert instance._serialize_fields_document() == snapshots[name]
        assert instance.marker() == methods[name]
    for name,module in modules.items():
        assert sys.modules[name] is module


@pytest.mark.parametrize("cross_file", [False,True])
def test_actual_base_change_rejects_whole_batch_without_touching_live_objects(scene, project, cross_file):
    paths = write_tree(project, cross_file, "old")
    types = {cls.__name__:cls for path in paths for cls in load_all_components_from_file(str(path), source_only=True)}
    instance = scene.create_game_object("Strict hierarchy").add_component(types["Leaf"])
    instance.value = 37
    before = instance._serialize_fields_document()
    marker = instance.marker()
    write_tree(project, cross_file, "new", changed_base=True)
    with pytest.raises(ScriptReloadRejected, match="base classes changed"):
        stage_component_body_reload_batch(tuple(ComponentBodyReloadRequest(str(path)) for path in paths))
    assert instance._serialize_fields_document() == before
    assert instance.marker() == marker


@pytest.mark.parametrize("cross_file", [False,True])
def test_publication_failure_restores_candidate_bases_and_live_schema(scene, project, monkeypatch, cross_file):
    from infernux.components import registry

    paths = write_tree(project, cross_file, "old")
    types = {cls.__name__:cls for path in paths for cls in load_all_components_from_file(str(path), source_only=True)}
    instance = scene.create_game_object("Rollback hierarchy").add_component(types["Leaf"])
    instance.value = 43
    before = instance._serialize_fields_document()
    write_tree(project, cross_file, "new", add_field=True)
    transaction = stage_component_body_reload_batch(tuple(ComponentBodyReloadRequest(str(path)) for path in paths))
    candidates = {cls: cls.__bases__ for _,classes in transaction.registry_entries for cls in classes
                  if cls.__name__ in {"Unmounted", "Added"}}
    calls = []

    def reject_publication(*args, **kwargs):
        calls.append(True)
        assert instance.marker().startswith("new:Base")
        assert instance.extra == 9
        assert all(cls.__bases__ == (types["Base"],) for cls in candidates)
        raise RuntimeError("reject inherited registry publication")

    monkeypatch.setattr(registry, "publish_component_script_types_batch", reject_publication)
    with pytest.raises(RuntimeError, match="reject inherited registry publication"):
        transaction.commit()
    assert calls == [True]
    assert transaction.rolled_back
    assert {cls: cls.__bases__ for cls in candidates} == candidates
    assert instance._serialize_fields_document() == before
    assert instance.marker() == "old:Base:old:Child:old:Leaf"
    assert "extra" not in get_serialized_fields(type(instance))
