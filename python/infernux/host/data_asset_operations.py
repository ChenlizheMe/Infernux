"""DataAsset operations projected from the engine authoring transaction."""

from __future__ import annotations

from .editor import EditorAutomationHost
from .operations import Operation, OperationError, OperationKind

from .operation_support import asset_path, on_editor, operation, set_json_pointer


def build_data_asset_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.data_asset.schema",
            OperationKind.QUERY,
            "Describe one DataAsset's authoritative serialized fields by asset GUID.",
            _data_asset_schema,
            capability="asset.read",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            tags=("data-asset", "asset", "guid", "schema"),
        ),
        operation(
            "infernux.data_asset.inspect",
            OperationKind.QUERY,
            "Read one DataAsset's canonical document by asset GUID.",
            _inspect_data_asset,
            capability="asset.read",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            tags=("data-asset", "asset", "guid", "inspect"),
        ),
        operation(
            "infernux.data_asset.property.set",
            OperationKind.COMMAND,
            "Set an existing DataAsset field through the engine document transaction.",
            _set_data_asset_property,
            capability="asset.write",
            input_properties={
                "asset_guid": {"type": "string"},
                "pointer": {"type": "string"},
                "value": {},
            },
            required=("asset_guid", "pointer", "value"),
            side_effects=("Changes and durably saves a DataAsset through editor history.",),
            reversible=True,
            tags=("data-asset", "asset", "guid", "property", "authoring"),
        ),
    )


def _load_data_asset(asset_guid: str):
    path = asset_path(asset_guid, suffix=".inxdata")
    asset, document = EditorAutomationHost.instance().data_asset_document(path)
    return path, asset, document


def _inspect_data_asset(asset_guid: str) -> dict[str, object]:
    def read():
        path, _asset, document = _load_data_asset(asset_guid)
        return {"asset_guid": asset_guid, "path": path, "document": document}

    return on_editor("infernux.data_asset.inspect", read)


def _data_asset_schema(asset_guid: str) -> dict[str, object]:
    def read():
        path = asset_path(asset_guid, suffix=".inxdata")
        return {
            "asset_guid": asset_guid,
            "path": path,
            **EditorAutomationHost.instance().data_asset_schema(path),
        }

    return on_editor("infernux.data_asset.schema", read)


def _set_data_asset_property(
    asset_guid: str,
    pointer: str,
    value,
) -> dict[str, object]:
    def edit():
        normalized_pointer = str(pointer or "").strip()
        if not normalized_pointer.startswith("/fields/"):
            raise OperationError(
                "operation.invalid_arguments",
                "DataAsset edits must address an existing /fields/... value.",
            )
        path, _asset, before = _load_data_asset(asset_guid)
        after = set_json_pointer(before, normalized_pointer, value)
        host = EditorAutomationHost.instance()
        host.publish_data_asset_document(
            path,
            asset_guid,
            after,
            edit_key=f"data_asset:{normalized_pointer}",
            description=f"Set DataAsset {normalized_pointer}",
        )
        _asset, published = host.data_asset_document(path)
        return {
            "asset_guid": asset_guid,
            "path": path,
            "pointer": normalized_pointer,
            "document": published,
        }

    return on_editor("infernux.data_asset.property.set", edit)


__all__ = ["build_data_asset_operations"]
