"""Explicit derived exclusions govern metadata, storage and saved documents."""
from typing import Annotated, ClassVar

import pytest

from infernux.components import InxComponent, SerializableObject, NonSerialized, hide_field, serialized_field
from infernux.components.fields import get_serialized_fields


def make_type(name, base, values):
    return type(name, (base,), {"__module__":__name__, **values})


def excluded_declaration(kind):
    if kind == "hidden":
        return {"transient":hide_field(6)}
    if kind == "nonserialized":
        return {"__annotations__":{"transient":Annotated[int,NonSerialized]},"transient":6}
    if kind == "descriptor":
        return {"__annotations__":{"transient":Annotated[int,NonSerialized]},"transient":serialized_field(6)}
    if kind == "annotation":
        return {"__annotations__":{"transient":Annotated[int,NonSerialized]}}
    return {"__annotations__":{"transient":ClassVar[int]},"transient":6}


@pytest.mark.parametrize("owner", [InxComponent,SerializableObject])
@pytest.mark.parametrize("kind", ["hidden","nonserialized","descriptor","annotation","classvar"])
@pytest.mark.parametrize("depth", [1,2])
def test_excluded_inherited_field_stays_runtime_only(scene, owner, kind, depth):
    prefix = f"{owner.__name__}_{kind}_{depth}"
    base = make_type(prefix+"Base",owner,{"transient":5,"stable":23})
    derived = make_type(prefix+"Derived",base,excluded_declaration(kind))
    cls = make_type(prefix+"Leaf",derived,{}) if depth == 2 else derived
    instance = scene.create_game_object(prefix).add_component(cls) if owner is InxComponent else cls()
    assert instance.transient == (5 if kind == "annotation" else 6)
    instance.transient = 37
    instance.stable = 41
    assert instance.transient == 37
    assert tuple(get_serialized_fields(cls)) == ("stable",)
    assert tuple(cls._field_schemas_) == ("stable",)
    if owner is InxComponent:
        from infernux.components import _cds_bridge
        assert set(_cds_bridge.get_class_info(cls)[1]) == {"stable"}
        fields = instance._serialize_fields_document()
        assert "transient" not in instance.game_object.serialize_document()["components"][0]["data"]
    else:
        fields = instance._serialize()["fields"]
        restored = SerializableObject._deserialize(instance._serialize())
        assert restored.stable == 41
        assert restored.transient != 37
    assert "transient" not in fields
    assert fields["stable"] == 41


@pytest.mark.parametrize("owner", [InxComponent,SerializableObject])
@pytest.mark.parametrize("kind", ["hidden","nonserialized"])
def test_more_derived_serialized_declaration_restores_field(scene, owner, kind):
    prefix = f"Restored_{owner.__name__}_{kind}"
    base = make_type(prefix+"Base",owner,{"transient":5})
    hidden = make_type(prefix+"Hidden",base,excluded_declaration(kind))
    cls = make_type(prefix+"Leaf",hidden,{"transient":9})
    instance = scene.create_game_object(prefix).add_component(cls) if owner is InxComponent else cls()
    instance.transient = 37
    fields = instance._serialize_fields_document() if owner is InxComponent else instance._serialize()["fields"]
    assert fields["transient"] == 37
    assert set(cls._field_schemas_) == {"transient"}


@pytest.mark.parametrize("owner", [InxComponent,SerializableObject])
def test_excluded_base_field_does_not_claim_a_serialized_identity(scene, owner):
    prefix = f"ExcludedId_{owner.__name__}"
    base = make_type(prefix+"Base",owner,{"transient":serialized_field(5,field_id="slot")})
    cls = make_type(prefix+"Child",base,{
        "transient":hide_field(6), "persistent":serialized_field(12,field_id="slot"),
    })
    instance = scene.create_game_object(prefix).add_component(cls) if owner is InxComponent else cls()
    assert instance.transient == 6
    assert instance.persistent == 12
    assert set(get_serialized_fields(cls)) == {"persistent"}


@pytest.mark.parametrize("owner", [InxComponent,SerializableObject])
@pytest.mark.parametrize("kind", ["hidden","nonserialized","descriptor","annotation","classvar"])
def test_explicit_private_base_field_can_be_excluded(scene, owner, kind):
    prefix = f"Private_{owner.__name__}_{kind}"
    base = make_type(prefix+"Base",owner,{"_transient":serialized_field(5),"stable":23})
    values = {("_transient" if name == "transient" else name):value
              for name,value in excluded_declaration(kind).items()}
    if "__annotations__" in values:
        values["__annotations__"] = {"_transient":values["__annotations__"]["transient"]}
    cls = make_type(prefix+"Child",base,values)
    instance = scene.create_game_object(prefix).add_component(cls) if owner is InxComponent else cls()
    assert instance._transient == (5 if kind == "annotation" else 6)
    instance._transient = 37
    assert instance._transient == 37
    assert set(get_serialized_fields(cls)) == {"stable"}
    assert set(cls._field_schemas_) == {"stable"}


@pytest.mark.parametrize("owner", [InxComponent,SerializableObject])
@pytest.mark.parametrize("hidden_first", [True,False])
def test_multiple_inheritance_uses_the_most_derived_mro_declaration(scene, owner, hidden_first):
    prefix = f"Diamond_{owner.__name__}_{hidden_first}"
    base = make_type(prefix+"Base",owner,{"transient":5})
    left = make_type(prefix+"Left",base,{"transient":hide_field(6)})
    right = make_type(prefix+"Right",base,{"transient":9})
    cls = type(prefix+"Leaf",(left,right) if hidden_first else (right,left),{"__module__":__name__})
    instance = scene.create_game_object(prefix).add_component(cls) if owner is InxComponent else cls()
    instance.transient = 37
    fields = instance._serialize_fields_document() if owner is InxComponent else instance._serialize()["fields"]
    assert ("transient" in fields) is not hidden_first
    if not hidden_first:
        assert fields["transient"] == 37


@pytest.mark.parametrize("declaration", ["transient = hide_field(6)","transient: Annotated[int, NonSerialized] = 6"])
def test_hot_reload_can_exclude_restore_and_rollback_inherited_fields(scene, tmp_path, declaration):
    import sys
    from infernux.components import registry, _cds_bridge
    from infernux.components.script_loader import ComponentBodyReloadRequest, load_all_components_from_file, stage_component_body_reload_batch
    from infernux.engine.project_context import get_project_root, set_project_root

    previous_root = get_project_root()
    registry_before = registry.snapshot_component_registry_state()
    module_before = sys.modules.get("mask_reload")
    assets = tmp_path / "Assets"
    assets.mkdir()
    path = assets / "mask_reload.py"
    set_project_root(str(tmp_path))

    def write(field):
        path.write_text(
            "from typing import Annotated\n"
            "from infernux.components import InxComponent, NonSerialized, hide_field\n"
            "class Base(InxComponent):\n    stable: int = 23\n    transient: int = 5\n"
            f"class Derived(Base):\n    {field}\n", encoding="utf-8",
        )

    transactions = []
    try:
        write("transient: int = 7")
        _,cls = load_all_components_from_file(str(path),source_only=True)
        instance = scene.create_game_object("Mask reload").add_component(cls)
        instance.stable,instance.transient = 41,37
        before = instance._serialize_fields_document()
        write(declaration)
        first = stage_component_body_reload_batch((ComponentBodyReloadRequest(str(path)),))
        transactions.append(first)
        first.commit()
        assert type(instance) is cls
        assert instance.stable == 41
        assert instance.transient == 6
        assert set(_cds_bridge.get_class_info(cls)[1]) == {"stable"}
        assert "transient" not in instance._serialize_fields_document()
        first.rollback()
        assert instance._serialize_fields_document() == before
        assert instance.transient == 37
        durable = stage_component_body_reload_batch((ComponentBodyReloadRequest(str(path)),))
        transactions.append(durable)
        durable.commit()
        durable.finalize()
        instance.transient = 52
        hidden_snapshot = instance._serialize_fields_document()

        write("transient: int = 9")
        second = stage_component_body_reload_batch((ComponentBodyReloadRequest(str(path)),))
        transactions.append(second)
        second.commit()
        assert instance.stable == 41
        assert instance.transient == 9
        assert set(_cds_bridge.get_class_info(cls)[1]) == {"stable","transient"}
        second.rollback()
        assert instance._serialize_fields_document() == hidden_snapshot
        assert instance.transient == 52
    finally:
        for transaction in reversed(transactions):
            if not transaction.finalized:
                transaction.rollback()
        registry.restore_component_registry_state(registry_before)
        if module_before is None:
            sys.modules.pop("mask_reload",None)
        else:
            sys.modules["mask_reload"] = module_before
        set_project_root(previous_root)
