"""Initial script loading owns data identities as strictly as hot reload."""
import sys
import types

import pytest

from infernux.components import SerializableObject
from infernux.components import registry, serializable_object
from infernux.components.script_loader import (
    get_script_error_by_path, load_all_components_from_file, load_and_create_component,
    _snapshot_script_diagnostics, _restore_script_diagnostics,
)
from infernux.engine.project_context import get_project_root, set_project_root


@pytest.fixture
def project(tmp_path):
    previous_root = get_project_root()
    previous_types = dict(serializable_object._SERIALIZABLE_REGISTRY)
    previous_components = registry.snapshot_component_registry_state()
    previous_diagnostics = _snapshot_script_diagnostics()
    names = ("initial_data_owner", "initial_data_other")
    previous_modules = {name:sys.modules.get(name) for name in names}
    assets = tmp_path / "Assets"
    assets.mkdir()
    set_project_root(str(tmp_path))
    try:
        yield assets
    finally:
        serializable_object._SERIALIZABLE_REGISTRY.clear()
        serializable_object._SERIALIZABLE_REGISTRY.update(previous_types)
        registry.restore_component_registry_state(previous_components)
        _restore_script_diagnostics(previous_diagnostics)
        for name,module in previous_modules.items():
            if module is None:
                sys.modules.pop(name,None)
            else:
                sys.modules[name] = module
        set_project_root(previous_root)


def source(identity, *, health=True, extra=""):
    return (
        "from infernux.components import SerializableObject, InxComponent\n"
        "class Payload(SerializableObject):\n"
        f"    __serialized_type_id__ = '{identity}'\n"
        + ("    health: int = 10\n" if health else "    label: str = 'new default'\n")
        + "class Carrier(InxComponent):\n    payload: Payload\n"
        + extra
    )


def load(path, entry):
    if entry == "all":
        return load_all_components_from_file(str(path), source_only=True)
    instance = load_and_create_component(str(path), script_guid=("a" if path.stem.endswith("owner") else "b") * 32)
    if instance is not None:
        instance._call_on_destroy()
    return instance


@pytest.mark.parametrize("entry", ["all", "create"])
@pytest.mark.parametrize("collision", [False, True])
def test_initial_load_preserves_other_modules_data_identity(project, entry, collision):
    first = project / "initial_data_owner.py"
    second = project / "initial_data_other.py"
    first.write_text(source("initial-test:payload"),encoding="utf-8")
    assert load(first,entry)
    owner = serializable_object.get_serializable_class("initial-test:payload")
    document = owner(health=87)._serialize()
    second.write_text(source("initial-test:payload" if collision else "initial-test:other",health=False),encoding="utf-8")
    result = load(second,entry)
    if collision:
        assert not result
        assert "owned by module" in get_script_error_by_path(str(second))
        assert not registry.component_types_for_script_path(str(second))
        assert "initial_data_other" not in sys.modules
    else:
        assert result
        assert serializable_object.get_serializable_class("initial-test:other") is not owner
    restored = SerializableObject._deserialize(document)
    assert type(restored) is owner
    assert restored.health == 87
    assert restored._serialize() == document


@pytest.mark.parametrize("existing", [False,True])
def test_failed_execution_publishes_neither_data_nor_component_prefix(project, existing):
    path = project / "initial_data_owner.py"
    if existing:
        path.write_text(source("initial-test:payload"),encoding="utf-8")
        assert load(path,"all")
    old_module = sys.modules.get("initial_data_owner")
    before_data = dict(serializable_object.get_registered_serializable_types())
    before_components = registry.component_types_for_script_path(str(path))
    path.write_text(source("initial-test:payload",health=False,extra="raise RuntimeError('source execution failed')\n"),encoding="utf-8")
    assert not load(path,"all")
    assert "source execution failed" in get_script_error_by_path(str(path))
    assert dict(serializable_object.get_registered_serializable_types()) == before_data
    assert registry.component_types_for_script_path(str(path)) == before_components
    assert sys.modules.get("initial_data_owner") is old_module


def test_successful_revision_replaces_owned_types_and_retires_deleted_types(project):
    path = project / "initial_data_owner.py"
    path.write_text(source("initial-test:payload"),encoding="utf-8")
    assert load(path,"all")
    old_type = serializable_object.get_serializable_class("initial-test:payload")
    document = old_type(health=87)._serialize()
    path.write_text(source("initial-test:payload").replace("= 10","= 20"),encoding="utf-8")
    assert load(path,"all")
    current = serializable_object.get_serializable_class("initial-test:payload")
    assert current is not old_type
    assert current().health == 20
    assert SerializableObject._deserialize(document).health == 87
    path.write_text("from infernux.components import InxComponent\nclass Empty(InxComponent): pass\n",encoding="utf-8")
    assert load(path,"all")
    assert serializable_object.get_serializable_class("initial-test:payload") is None


def test_ordinary_module_declaration_cannot_steal_another_owner(project):
    first = types.ModuleType("initial_data_owner")
    second = types.ModuleType("initial_data_other")
    sys.modules[first.__name__] = first
    sys.modules[second.__name__] = second
    declaration = "from infernux.components import SerializableObject\nclass Payload(SerializableObject):\n    __serialized_type_id__ = 'initial-test:ordinary'\n    health: int = 12\n"
    exec(declaration,first.__dict__)
    with pytest.raises(ValueError,match="owned by module"):
        exec(declaration,second.__dict__)
    assert serializable_object.get_serializable_class("initial-test:ordinary") is first.Payload


@pytest.mark.parametrize("register", [False,True])
def test_module_level_instances_keep_native_storage_and_dispatch(scene, project, register):
    from infernux.engine.runtime_dispatch import current_runtime_epoch
    path = project / "initial_data_owner.py"
    path.write_text(
        "from infernux.components import InxComponent\n"
        "class Carrier(InxComponent):\n"
        "    value: int = 7\n"
        "    def update(self, dt): self.value += 1\n"
        "INSTANCE = Carrier()\nINSTANCE.value = 43\n", encoding="utf-8",
    )
    loaded = load_all_components_from_file(str(path),register=register,source_only=True)
    instance = sys.modules["initial_data_owner"].INSTANCE
    assert type(instance) is loaded[0]
    assert instance._cds_slot is not None
    assert instance._cds_class_id is not None
    assert current_runtime_epoch().descriptor_for(type(instance)) is not None
    scene.create_game_object("Initial singleton").add_py_component(instance)
    instance.update(0.1)
    assert instance.value == 44
    assert instance._serialize_fields_document()["value"] == 44


def test_failure_after_data_publication_restores_live_owner(scene, project, monkeypatch):
    from contextlib import contextmanager
    from infernux.engine.runtime_dispatch import current_runtime_epoch
    from infernux.renderstack import render_effect_compiler
    path = project / "initial_data_owner.py"
    path.write_text(source("initial-test:payload"),encoding="utf-8")
    old_type, = load_all_components_from_file(str(path),source_only=True)
    old_module = sys.modules["initial_data_owner"]
    instance = scene.create_game_object("Retained data").add_component(old_type)
    instance.payload.health = 87
    before = instance._serialize_fields_document()
    path.write_text(source("initial-test:payload",health=False),encoding="utf-8")
    candidates = []

    @contextmanager
    def reject_effect_commit(_path):
        yield
        module = sys.modules["initial_data_owner"]
        candidates.append(module.Carrier)
        assert serializable_object.get_serializable_class("initial-test:payload") is module.Payload
        raise RuntimeError("effect publication failure")

    monkeypatch.setattr(render_effect_compiler,"_stage_source_effect_features",reject_effect_commit)
    assert load_all_components_from_file(str(path),source_only=True) == []
    assert "effect publication failure" in get_script_error_by_path(str(path))
    assert sys.modules["initial_data_owner"] is old_module
    assert serializable_object.get_serializable_class("initial-test:payload") is old_module.Payload
    assert registry.component_types_for_script_path(str(path)) == (old_type,)
    assert instance._serialize_fields_document() == before
    assert type(instance) is old_type
    assert current_runtime_epoch().descriptor_for(old_type) is not None
    assert len(candidates) == 1
    assert current_runtime_epoch().descriptor_for(candidates[0]) is None


def test_failed_owner_keeps_successful_imported_module_publication(project):
    helper = project / "initial_data_other.py"
    helper.write_text(source("initial-test:helper"),encoding="utf-8")
    path = project / "initial_data_owner.py"
    path.write_text(source("initial-test:payload",extra="import initial_data_other\nraise RuntimeError('owner failed')\n"),encoding="utf-8")
    assert load_all_components_from_file(str(path),source_only=True) == []
    helper_module = sys.modules["initial_data_other"]
    assert serializable_object.get_serializable_class("initial-test:helper") is helper_module.Payload
    assert serializable_object.get_serializable_class("initial-test:payload") is None
    assert registry.component_types_for_script_path(str(helper)) == (helper_module.Carrier,)
    assert not registry.component_types_for_script_path(str(path))
