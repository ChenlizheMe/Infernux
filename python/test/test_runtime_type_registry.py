from __future__ import annotations

import json

import pytest


def _semantic(script_guid: str, type_guid: str, readable_id: str, lifecycle=()):
    return {
        "type_guid": type_guid,
        "readable_id": readable_id,
        "owner": f"script:{script_guid}",
        "origin": "python",
        "schema_version": 1,
        "display_name": readable_id,
        "base_type_guid": "",
        "constructible": True,
        "serializable": True,
        "runtime_available": True,
        "runtime_profiles": ["player"],
        "lifecycle": list(lifecycle),
        "fields": [],
    }


def test_runtime_type_registry_binds_declared_phase_contract(tmp_path):
    from Infernux.components.component import InxComponent
    from Infernux.engine.runtime_type_registry import (
        bind_runtime_lifecycle_contract,
        clear_runtime_type_registry,
        install_runtime_type_registry,
        validate_runtime_component_identity,
    )

    class PlayerMover(InxComponent):
        def update(self, delta_time):
            del delta_time

    type_guid = PlayerMover._get_type_guid()
    path = tmp_path / "RuntimeTypeRegistry.json"
    path.write_text(
        json.dumps(
            {
                "$schema": "infernux.runtime_type_registry",
                "types": [
                    {
                        "script_guid": "script-guid",
                        "type_guid": type_guid,
                        "type_id": "test.PlayerMover",
                        "module": PlayerMover.__module__,
                        "qualname": PlayerMover.__qualname__,
                        "runtime_path": "Assets/Scripts/player.pyc",
                        "lifecycle": ["update"],
                        "semantic": _semantic(
                            "script-guid", type_guid, "test.PlayerMover", ["update"]
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        assert install_runtime_type_registry(str(path)) == 1
        from Infernux.lib import _Infernux as native

        descriptor = native._semantic_catalog_snapshot().type_document(type_guid)
        assert descriptor["owner"] == "script:script-guid"
        assert descriptor["runtime_profiles"] == ["player"]
        record = validate_runtime_component_identity(
            script_guid="script-guid",
            type_guid=type_guid,
            module_name=PlayerMover.__module__,
            qualified_name=PlayerMover.__qualname__,
        )
        bind_runtime_lifecycle_contract(PlayerMover, record)
    finally:
        clear_runtime_type_registry()

    with pytest.raises(KeyError, match=type_guid):
        native._semantic_catalog_snapshot().type_document(type_guid)


def test_runtime_type_registry_requires_cooked_semantics(tmp_path):
    from Infernux.engine.runtime_type_registry import install_runtime_type_registry

    path = tmp_path / "RuntimeTypeRegistry.json"
    path.write_text(
        json.dumps(
            {
                "$schema": "infernux.runtime_type_registry",
                "types": [
                    {
                        "script_guid": "script",
                        "type_guid": "type",
                        "type_id": "test.Player",
                        "module": "Scripts.player",
                        "qualname": "Player",
                        "runtime_path": "Assets/Scripts/player.pyc",
                        "lifecycle": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no cooked semantic descriptor"):
        install_runtime_type_registry(str(path))


def test_runtime_type_registry_rejects_unlisted_component(tmp_path):
    from Infernux.engine.runtime_type_registry import (
        clear_runtime_type_registry,
        install_runtime_type_registry,
        validate_runtime_component_identity,
    )

    path = tmp_path / "RuntimeTypeRegistry.json"
    path.write_text(
        json.dumps(
            {
                "$schema": "infernux.runtime_type_registry",
                "types": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        install_runtime_type_registry(str(path))
        # An empty registry is valid, but no user component may appear later.
        with pytest.raises(RuntimeError, match="absent"):
            validate_runtime_component_identity(
                script_guid="script",
                type_guid="missing",
                module_name="Scripts.missing",
                qualified_name="Missing",
            )
    finally:
        clear_runtime_type_registry()


def test_runtime_type_registry_rejects_identity_drift(tmp_path):
    from Infernux.engine.runtime_type_registry import (
        clear_runtime_type_registry,
        install_runtime_type_registry,
        validate_runtime_component_identity,
    )

    path = tmp_path / "RuntimeTypeRegistry.json"
    path.write_text(
        json.dumps(
            {
                "$schema": "infernux.runtime_type_registry",
                "types": [
                    {
                        "script_guid": "script",
                        "type_guid": "type",
                        "module": "Scripts.player",
                        "qualname": "Player",
                        "runtime_path": "Assets/Scripts/player.pyc",
                        "lifecycle": [],
                        "semantic": _semantic("script", "type", "test.Player"),
                        "type_id": "test.Player",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    try:
        install_runtime_type_registry(str(path))
        with pytest.raises(RuntimeError, match="disagrees"):
            validate_runtime_component_identity(
                script_guid="script",
                type_guid="type",
                module_name="Scripts.renamed",
                qualified_name="Player",
            )
    finally:
        clear_runtime_type_registry()
