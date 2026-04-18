"""
NodeGraph — Generic visual node-graph data model.

Reusable foundation for any node-based editor (state machines,
shader graphs, dialogue trees, etc.).  The *view* layer lives in
:mod:`Infernux.engine.ui.node_graph_view`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
import json
import uuid


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class PinKind(IntEnum):
    INPUT = 0
    OUTPUT = 1


# ═══════════════════════════════════════════════════════════════════════════
# Definitions (registered once per node type)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PinDef:
    """Definition of a single pin on a node type."""

    id: str
    label: str
    kind: PinKind
    color: tuple = (0.80, 0.80, 0.80, 1.0)
    max_connections: int = -1  # -1 = unlimited


@dataclass
class NodeTypeDef:
    """Registered blueprint for a category of nodes."""

    type_id: str
    label: str
    header_color: tuple = (0.30, 0.30, 0.30, 1.0)
    pins: List[PinDef] = field(default_factory=list)
    min_width: float = 140.0
    deletable: bool = True
    body_bottom_pad: float = 0.0  # extra height below pins for custom body UI (px at zoom=1)

    def input_pins(self) -> List[PinDef]:
        return [p for p in self.pins if p.kind == PinKind.INPUT]

    def output_pins(self) -> List[PinDef]:
        return [p for p in self.pins if p.kind == PinKind.OUTPUT]


# ═══════════════════════════════════════════════════════════════════════════
# Instances (per-graph objects)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """A concrete node placed on the canvas."""

    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    type_id: str = ""
    pos_x: float = 0.0
    pos_y: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "type_id": self.type_id,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "data": dict(self.data),
        }

    @staticmethod
    def from_dict(d: dict) -> GraphNode:
        return GraphNode(
            uid=d["uid"],
            type_id=d["type_id"],
            pos_x=d.get("pos_x", 0.0),
            pos_y=d.get("pos_y", 0.0),
            data=d.get("data", {}),
        )


@dataclass
class GraphLink:
    """A directed connection between two pins on two nodes."""

    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_node: str = ""   # node uid
    source_pin: str = ""    # pin id
    target_node: str = ""   # node uid
    target_pin: str = ""    # pin id
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "source_node": self.source_node,
            "source_pin": self.source_pin,
            "target_node": self.target_node,
            "target_pin": self.target_pin,
            "data": dict(self.data),
        }

    @staticmethod
    def from_dict(d: dict) -> GraphLink:
        return GraphLink(
            uid=d["uid"],
            source_node=d["source_node"],
            source_pin=d["source_pin"],
            target_node=d["target_node"],
            target_pin=d["target_pin"],
            data=d.get("data", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════

class NodeGraph:
    """Generic node-graph container with CRUD and serialisation."""

    def __init__(self) -> None:
        self.nodes: List[GraphNode] = []
        self.links: List[GraphLink] = []
        self._type_registry: Dict[str, NodeTypeDef] = {}

    # ── Type registry ─────────────────────────────────────────────────

    def register_type(self, typedef: NodeTypeDef) -> None:
        self._type_registry[typedef.type_id] = typedef

    def get_type(self, type_id: str) -> Optional[NodeTypeDef]:
        return self._type_registry.get(type_id)

    def registered_types(self) -> List[NodeTypeDef]:
        return list(self._type_registry.values())

    # ── Node CRUD ─────────────────────────────────────────────────────

    def add_node(
        self,
        type_id: str,
        x: float = 0.0,
        y: float = 0.0,
        uid: Optional[str] = None,
        **data: Any,
    ) -> GraphNode:
        node = GraphNode(
            uid=uid or uuid.uuid4().hex[:8],
            type_id=type_id,
            pos_x=x,
            pos_y=y,
            data=data,
        )
        self.nodes.append(node)
        return node

    def remove_node(self, uid: str) -> bool:
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.uid != uid]
        self.links = [
            lk for lk in self.links
            if lk.source_node != uid and lk.target_node != uid
        ]
        return len(self.nodes) < before

    def find_node(self, uid: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.uid == uid:
                return n
        return None

    # ── Link CRUD ─────────────────────────────────────────────────────

    def add_link(
        self,
        src_node: str,
        src_pin: str,
        dst_node: str,
        dst_pin: str,
        uid: Optional[str] = None,
        **data: Any,
    ) -> Optional[GraphLink]:
        # Disallow self-loops
        if src_node == dst_node:
            return None
        # Disallow duplicates
        for lk in self.links:
            if (lk.source_node == src_node and lk.source_pin == src_pin
                    and lk.target_node == dst_node and lk.target_pin == dst_pin):
                return None
        link = GraphLink(
            uid=uid or uuid.uuid4().hex[:8],
            source_node=src_node,
            source_pin=src_pin,
            target_node=dst_node,
            target_pin=dst_pin,
            data=data,
        )
        self.links.append(link)
        return link

    def remove_link(self, uid: str) -> bool:
        before = len(self.links)
        self.links = [lk for lk in self.links if lk.uid != uid]
        return len(self.links) < before

    def find_link(self, uid: str) -> Optional[GraphLink]:
        for lk in self.links:
            if lk.uid == uid:
                return lk
        return None

    def get_links_for_node(self, node_uid: str) -> List[GraphLink]:
        return [
            lk for lk in self.links
            if lk.source_node == node_uid or lk.target_node == node_uid
        ]

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [lk.to_dict() for lk in self.links],
        }

    def load_dict(self, d: dict) -> None:
        self.nodes = [GraphNode.from_dict(nd) for nd in d.get("nodes", [])]
        self.links = [GraphLink.from_dict(lk) for lk in d.get("links", [])]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def load_json(self, text: str) -> None:
        self.load_dict(json.loads(text))

    def clear(self) -> None:
        self.nodes.clear()
        self.links.clear()
