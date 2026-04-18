"""Unit tests for :mod:`Infernux.core.node_graph` (pure Python, no native engine)."""
from __future__ import annotations

import pytest

from Infernux.core.node_graph import (
    SCHEMA_VERSION,
    GraphLink,
    GraphNode,
    NodeGraph,
    NodeTypeDef,
    PinDef,
    PinKind,
)


def _sample_registry(g: NodeGraph) -> tuple[GraphNode, GraphNode]:
    g.register_type(
        NodeTypeDef(
            type_id="src",
            label="Source",
            pins=[
                PinDef("out", "Out", PinKind.OUTPUT, max_connections=1),
                PinDef("in_bad", "In", PinKind.INPUT),
            ],
        )
    )
    g.register_type(
        NodeTypeDef(
            type_id="dst",
            label="Dest",
            pins=[
                PinDef("in", "In", PinKind.INPUT, max_connections=1),
                PinDef("out_bad", "Out", PinKind.OUTPUT),
            ],
        )
    )
    a = g.add_node("src", 0, 0, uid="n_src")
    b = g.add_node("dst", 100, 0, uid="n_dst")
    return a, b


def test_add_node_requires_registered_type():
    g = NodeGraph()
    with pytest.raises(ValueError, match="Unknown node type_id"):
        g.add_node("missing")


def test_to_dict_includes_schema_version():
    g = NodeGraph()
    _sample_registry(g)
    d = g.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert "nodes" in d and "links" in d


def test_add_link_output_to_input_only():
    g = NodeGraph()
    a, b = _sample_registry(g)
    assert g.add_link(a.uid, "out", b.uid, "in") is not None
    assert g.add_link(b.uid, "in", a.uid, "out") is None
    assert g.add_link(a.uid, "in_bad", b.uid, "in") is None


def test_add_link_rejects_unknown_pins_or_nodes():
    g = NodeGraph()
    a, b = _sample_registry(g)
    assert g.add_link(a.uid, "no_such", b.uid, "in") is None
    assert g.add_link("x", "out", b.uid, "in") is None


def test_add_link_rejects_duplicate():
    g = NodeGraph()
    a, b = _sample_registry(g)
    assert g.add_link(a.uid, "out", b.uid, "in") is not None
    assert g.add_link(a.uid, "out", b.uid, "in") is None


def test_add_link_respects_max_connections():
    g = NodeGraph()
    a, b = _sample_registry(g)
    c = g.add_node("dst", 200, 0, uid="n_dst2")
    assert g.add_link(a.uid, "out", b.uid, "in") is not None
    assert g.add_link(a.uid, "out", c.uid, "in") is None

    g2 = NodeGraph()
    a2, b2 = _sample_registry(g2)
    g2.add_node("src", 0, 50, uid="n_src2")
    n2 = g2.find_node("n_src2")
    assert n2 is not None
    assert g2.add_link(a2.uid, "out", b2.uid, "in") is not None
    assert g2.add_link(n2.uid, "out", b2.uid, "in") is None


def test_load_dict_prunes_unknown_types_and_orphan_links():
    g = NodeGraph()
    _sample_registry(g)
    g.load_dict(
        {
            "schema_version": 0,
            "nodes": [
                {"uid": "keep", "type_id": "src", "pos_x": 0, "pos_y": 0, "data": {}},
                {"uid": "gone", "type_id": "not_registered", "pos_x": 1, "pos_y": 1, "data": {}},
                {"uid": "dst_ok", "type_id": "dst", "pos_x": 2, "pos_y": 2, "data": {}},
            ],
            "links": [
                {
                    "uid": "l1",
                    "source_node": "keep",
                    "source_pin": "out",
                    "target_node": "dst_ok",
                    "target_pin": "in",
                    "data": {},
                },
                {
                    "uid": "l2",
                    "source_node": "gone",
                    "source_pin": "out",
                    "target_node": "dst_ok",
                    "target_pin": "in",
                    "data": {},
                },
            ],
        }
    )
    assert {n.uid for n in g.nodes} == {"keep", "dst_ok"}
    assert len(g.links) == 1
    assert g.links[0].uid == "l1"


def test_prune_invalid_drops_duplicate_link_signatures():
    g = NodeGraph()
    a, b = _sample_registry(g)
    g.links = [
        GraphLink(
            uid="x",
            source_node=a.uid,
            source_pin="out",
            target_node=b.uid,
            target_pin="in",
        ),
        GraphLink(
            uid="y",
            source_node=a.uid,
            source_pin="out",
            target_node=b.uid,
            target_pin="in",
        ),
    ]
    g._reindex_all()
    removed_n, removed_l = g.prune_invalid()
    assert removed_n == 0
    assert removed_l == 1
    assert len(g.links) == 1


def test_round_trip_dict_preserves_schema_version():
    g = NodeGraph()
    _sample_registry(g)
    g.add_link("n_src", "out", "n_dst", "in")
    blob = g.to_dict()
    g2 = NodeGraph()
    for td in g.registered_types():
        g2.register_type(td)
    g2.load_dict(blob)
    assert g2.to_dict()["schema_version"] == SCHEMA_VERSION
    assert len(g2.nodes) == 2
    assert len(g2.links) == 1
