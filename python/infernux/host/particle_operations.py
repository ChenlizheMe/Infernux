"""Strict Particle Graph asset operations with AOT validation."""

from __future__ import annotations

from infernux.host import EditorAutomationHost, Operation, OperationKind

from infernux.host.operation_support import asset_path, on_editor, operation, set_json_pointer


def build_particle_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.particle.graph.inspect",
            OperationKind.QUERY,
            "Read a strict Particle Graph document addressed by asset GUID.",
            _inspect_graph,
            capability="particle.read",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            tags=("particle", "graph", "asset", "guid", "inspect"),
        ),
        operation(
            "infernux.particle.graph.property.set",
            OperationKind.COMMAND,
            "Set an existing Particle Graph document value and AOT-validate the result.",
            _set_graph_property,
            capability="particle.write",
            input_properties={
                "asset_guid": {"type": "string"},
                "pointer": {"type": "string"},
                "value": {},
            },
            required=("asset_guid", "pointer", "value"),
            side_effects=("Compiles and durably saves a Particle Graph asset with Undo support.",),
            reversible=True,
            tags=("particle", "graph", "property", "aot", "authoring"),
        ),
        operation(
            "infernux.particle.graph.document.replace",
            OperationKind.COMMAND,
            "Replace a Particle Graph with a complete strict schema document.",
            _replace_graph_document,
            capability="particle.write",
            input_properties={
                "asset_guid": {"type": "string"},
                "document": {"type": "object"},
            },
            required=("asset_guid", "document"),
            side_effects=("Compiles and durably replaces a Particle Graph asset with Undo support.",),
            reversible=True,
            tags=("particle", "graph", "document", "aot", "authoring"),
        ),
    )


def _inspect_graph(asset_guid: str) -> dict[str, object]:
    def read():
        path = asset_path(asset_guid, suffix=".particlegraph")
        graph, document = EditorAutomationHost.instance().particle_graph_document(path)
        return {
            "asset_guid": asset_guid,
            "path": path,
            "semantic_hash": graph.semantic_hash(),
            "document": document,
        }

    return on_editor("infernux.particle.graph.inspect", read)


def _set_graph_property(asset_guid: str, pointer: str, value) -> dict[str, object]:
    def edit():
        path = asset_path(asset_guid, suffix=".particlegraph")
        host = EditorAutomationHost.instance()
        before, document = host.particle_graph_document(path)
        after = host.particle_graph_from_document(set_json_pointer(document, pointer, value))
        _commit_graph_change(path, asset_guid, before, after, f"Set Particle Graph {pointer}")
        return {
            "asset_guid": asset_guid,
            "path": path,
            "pointer": pointer,
            "semantic_hash": after.semantic_hash(),
            "document": after.to_dict(),
        }

    return on_editor("infernux.particle.graph.property.set", edit)


def _replace_graph_document(asset_guid: str, document: dict) -> dict[str, object]:
    def edit():
        path = asset_path(asset_guid, suffix=".particlegraph")
        host = EditorAutomationHost.instance()
        before, _document = host.particle_graph_document(path)
        after = host.particle_graph_from_document(document)
        _commit_graph_change(path, asset_guid, before, after, "Replace Particle Graph Document")
        return {
            "asset_guid": asset_guid,
            "path": path,
            "semantic_hash": after.semantic_hash(),
            "document": after.to_dict(),
        }

    return on_editor("infernux.particle.graph.document.replace", edit)


def _commit_graph_change(path: str, guid: str, before, after, description: str) -> None:
    EditorAutomationHost.instance().publish_particle_graph(
        path, guid, before, after, description
    )


__all__ = ["build_particle_operations"]
