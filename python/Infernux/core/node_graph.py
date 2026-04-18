"""
NodeGraph — Generic visual node-graph data model.

Reusable foundation for any node-based editor (state machines,
shader graphs, dialogue trees, etc.).  The *view* layer lives in
:mod:`Infernux.engine.ui.node_graph_view`.

Serialized JSON includes ``schema_version`` for forward-compatible migrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import json
import uuid

RGBA = Tuple[float, float, float, float]


SCHEMA_VERSION = 1


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
    # Header color swatch (click opens popup editor in :class:`~Infernux.engine.ui.node_graph_view.NodeGraphView`)
    header_color_swatch: bool = False
    # Pin row label tint; ``None`` uses ``Theme.NODE_GRAPH_TEXT_DIM``
    pin_label_color: Optional[RGBA] = None

    def input_pins(self) -> List[PinDef]:
        return [p for p in self.pins if p.kind == PinKind.INPUT]

    def output_pins(self) -> List[PinDef]:
        return [p for p in self.pins if p.kind == PinKind.OUTPUT]

    def pin_by_id(self, pin_id: str) -> Optional[PinDef]:
        for p in self.pins:
            if p.id == pin_id:
                return p
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Instances (per-graph objects)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """A concrete node placed on the canvas.

    Optional ``data`` keys consumed by :class:`~Infernux.engine.ui.node_graph_view.NodeGraphView`:

    - ``header_color``: ``[r, g, b]`` or ``[r, g, b, a]`` in 0..1 — overrides the type's header tint.
    """

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
        self._nodes_by_uid: Dict[str, GraphNode] = {}
        self._links_by_uid: Dict[str, GraphLink] = {}

    # ── Indices ─────────────────────────────────────────────────────

    def _reindex_nodes(self) -> None:
        self._nodes_by_uid = {n.uid: n for n in self.nodes}

    def _reindex_links(self) -> None:
        self._links_by_uid = {lk.uid: lk for lk in self.links}

    def _reindex_all(self) -> None:
        self._reindex_nodes()
        self._reindex_links()

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
        if type_id not in self._type_registry:
            raise ValueError(
                f"Unknown node type_id {type_id!r}; register it with register_type() before add_node()."
            )
        node = GraphNode(
            uid=uid or uuid.uuid4().hex[:8],
            type_id=type_id,
            pos_x=x,
            pos_y=y,
            data=data,
        )
        self.nodes.append(node)
        self._nodes_by_uid[node.uid] = node
        return node

    def remove_node(self, uid: str) -> bool:
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.uid != uid]
        self.links = [
            lk for lk in self.links
            if lk.source_node != uid and lk.target_node != uid
        ]
        if len(self.nodes) < before:
            self._nodes_by_uid.pop(uid, None)
            self._reindex_links()
            return True
        return False

    def find_node(self, uid: str) -> Optional[GraphNode]:
        return self._nodes_by_uid.get(uid)

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
        if src_node == dst_node:
            return None

        n_src = self.find_node(src_node)
        n_dst = self.find_node(dst_node)
        if n_src is None or n_dst is None:
            return None

        td_src = self.get_type(n_src.type_id)
        td_dst = self.get_type(n_dst.type_id)
        if td_src is None or td_dst is None:
            return None

        p_src = td_src.pin_by_id(src_pin)
        p_dst = td_dst.pin_by_id(dst_pin)
        if p_src is None or p_dst is None:
            return None
        if p_src.kind != PinKind.OUTPUT or p_dst.kind != PinKind.INPUT:
            return None

        for lk in self.links:
            if (lk.source_node == src_node and lk.source_pin == src_pin
                    and lk.target_node == dst_node and lk.target_pin == dst_pin):
                return None

        if p_src.max_connections >= 0:
            out_n = sum(
                1 for lk in self.links
                if lk.source_node == src_node and lk.source_pin == src_pin
            )
            if out_n >= p_src.max_connections:
                return None
        if p_dst.max_connections >= 0:
            in_n = sum(
                1 for lk in self.links
                if lk.target_node == dst_node and lk.target_pin == dst_pin
            )
            if in_n >= p_dst.max_connections:
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
        self._links_by_uid[link.uid] = link
        return link

    def remove_link(self, uid: str) -> bool:
        before = len(self.links)
        self.links = [lk for lk in self.links if lk.uid != uid]
        if len(self.links) < before:
            self._links_by_uid.pop(uid, None)
            return True
        return False

    def find_link(self, uid: str) -> Optional[GraphLink]:
        return self._links_by_uid.get(uid)

    def get_links_for_node(self, node_uid: str) -> List[GraphLink]:
        return [
            lk for lk in self.links
            if lk.source_node == node_uid or lk.target_node == node_uid
        ]

    # ── Validation (after load) ───────────────────────────────────────

    def _link_endpoints_valid(self, lk: GraphLink) -> bool:
        n_src = self.find_node(lk.source_node)
        n_dst = self.find_node(lk.target_node)
        if n_src is None or n_dst is None:
            return False
        td_src = self.get_type(n_src.type_id)
        td_dst = self.get_type(n_dst.type_id)
        if td_src is None or td_dst is None:
            return False
        p_src = td_src.pin_by_id(lk.source_pin)
        p_dst = td_dst.pin_by_id(lk.target_pin)
        if p_src is None or p_dst is None:
            return False
        if p_src.kind != PinKind.OUTPUT or p_dst.kind != PinKind.INPUT:
            return False
        return True

    def prune_invalid(self) -> Tuple[int, int]:
        """Drop unknown-type nodes and invalid/orphan links. Returns (nodes_removed, links_removed)."""
        known_types = set(self._type_registry.keys())
        before_n = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.type_id in known_types]
        nodes_removed = before_n - len(self.nodes)

        valid_uids = {n.uid for n in self.nodes}
        before_l = len(self.links)
        seen_sig: set[Tuple[str, str, str, str]] = set()
        kept: List[GraphLink] = []
        for lk in self.links:
            if lk.source_node not in valid_uids or lk.target_node not in valid_uids:
                continue
            if lk.source_node == lk.target_node:
                continue
            if not self._link_endpoints_valid(lk):
                continue
            sig = (lk.source_node, lk.source_pin, lk.target_node, lk.target_pin)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            kept.append(lk)

        out_counts: Dict[Tuple[str, str], int] = {}
        in_counts: Dict[Tuple[str, str], int] = {}
        filtered: List[GraphLink] = []
        for lk in kept:
            n_src = self.find_node(lk.source_node)
            n_dst = self.find_node(lk.target_node)
            if not n_src or not n_dst:
                continue
            td_src = self.get_type(n_src.type_id)
            td_dst = self.get_type(n_dst.type_id)
            if not td_src or not td_dst:
                continue
            ps = td_src.pin_by_id(lk.source_pin)
            pd = td_dst.pin_by_id(lk.target_pin)
            if not ps or not pd:
                continue
            k_out = (lk.source_node, lk.source_pin)
            k_in = (lk.target_node, lk.target_pin)
            if ps.max_connections >= 0 and out_counts.get(k_out, 0) >= ps.max_connections:
                continue
            if pd.max_connections >= 0 and in_counts.get(k_in, 0) >= pd.max_connections:
                continue
            if ps.max_connections >= 0:
                out_counts[k_out] = out_counts.get(k_out, 0) + 1
            if pd.max_connections >= 0:
                in_counts[k_in] = in_counts.get(k_in, 0) + 1
            filtered.append(lk)

        self.links = filtered
        links_removed = before_l - len(self.links)
        self._reindex_all()
        return nodes_removed, links_removed

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [lk.to_dict() for lk in self.links],
        }

    def load_dict(self, d: dict) -> None:
        self.nodes = [GraphNode.from_dict(nd) for nd in d.get("nodes", [])]
        self.links = [GraphLink.from_dict(lk) for lk in d.get("links", [])]
        self._reindex_all()
        # ``schema_version`` in *d* is reserved for future migrations.
        self.prune_invalid()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def load_json(self, text: str) -> None:
        self.load_dict(json.loads(text))

    def clear(self) -> None:
        self.nodes.clear()
        self.links.clear()
        self._nodes_by_uid.clear()
        self._links_by_uid.clear()
