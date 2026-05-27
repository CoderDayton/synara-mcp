"""Unit-level coverage of graph route helpers (caps, empties, overlays)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

import synara.features.dashboard.routes.graph as graph_mod


# ---------------------------------------------------------- fake collection
class _FakeEdge:
    def __init__(
        self,
        src: int,
        dst: int,
        *,
        hits: int = 1,
        weight: float = 0.0,
        bonus: float = 0.0,
    ) -> None:
        self.src_id = src
        self.dst_id = dst
        self.hits = hits
        self.weight = weight
        self.bonus = bonus


class _FakeColl:
    """Minimal stand-in for AsyncVectorCollection: stores edges + documents."""

    def __init__(
        self,
        sr_edges: list[_FakeEdge] | None = None,
        plast_edges: list[_FakeEdge] | None = None,
        documents: dict[int, tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._sr = sr_edges or []
        self._plast = plast_edges or []
        self._docs = documents or {}

    async def get_edges(
        self,
        *,
        kind: str,
        src: int | None = None,
        dst: int | None = None,
        limit: int | None = None,
    ) -> list[_FakeEdge]:
        pool = self._sr if kind == "sr" else self._plast
        out = [
            e for e in pool if (src is None or e.src_id == src) and (dst is None or e.dst_id == dst)
        ]
        if limit is not None:
            out = out[:limit]
        return out

    async def get_documents(
        self,
        filter_dict: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> Iterable[tuple[int, str, dict[str, Any]]]:
        if filter_dict is None:
            rows = [(doc_id, *doc) for doc_id, doc in self._docs.items()]
        else:
            ids = filter_dict.get("id", [])
            if not isinstance(ids, list):
                ids = [ids]
            rows = [(doc_id, *self._docs[doc_id]) for doc_id in ids if doc_id in self._docs]
        if limit is not None:
            rows = rows[:limit]
        return rows

    async def get_embeddings_by_ids(self, ids: list[int]) -> dict[int, Any]:
        return {}


# ---------------------------------------------------------- _global_sr cap
async def test_global_sr_skips_edges_when_cap_reached_and_src_not_seen() -> None:
    """Line 46: cap hit + src is a fresh node → edge dropped."""
    edges = [
        _FakeEdge(1, 2),
        _FakeEdge(3, 4),
        _FakeEdge(5, 6),  # past cap of 2; 5 is unknown → drop
    ]
    coll = _FakeColl(sr_edges=edges)
    nodes, kept = await graph_mod._global_sr(coll, max_nodes=2)
    # Cap = 2; first edge brings in {1, 2}, len(nodes) now == 2;
    # second edge: len == 2 and src 3 not in nodes → `continue` (line 46).
    assert 3 not in nodes
    assert 5 not in nodes
    assert kept[0]["src"] == 1


# ---------------------------------------------------------- _focus_neighborhood cap
async def test_focus_neighborhood_breaks_when_cap_reached_midfrontier() -> None:
    """Line 92: per-frontier `break` once the node cap is met."""
    # Anchor 0 has many SR successors — depth=1 BFS must stop once cap hit.
    edges = [_FakeEdge(0, k) for k in range(1, 6)]
    coll = _FakeColl(sr_edges=edges)
    nodes, _ = await graph_mod._focus_neighborhood(coll, focus=0, depth=2, max_nodes=3)
    assert 0 in nodes
    assert len(nodes) <= 3


# ---------------------------------------------------------- _preview
def test_preview_returns_empty_for_falsey_input() -> None:
    """Line 106: empty/None text → empty string."""
    assert graph_mod._preview("") == ""
    assert graph_mod._preview(None) == ""


def test_preview_truncates_and_appends_ellipsis() -> None:
    long = "x " * 200
    out = graph_mod._preview(long)
    assert out.endswith("…")
    assert len(out) <= graph_mod._PREVIEW_CHARS + 1


# ---------------------------------------------------------- _docs_by_id empty
async def test_docs_by_id_empty_short_circuits() -> None:
    """Line 120: empty id list → empty dict, no collection round-trip."""
    coll = _FakeColl(documents={1: ("a", {})})
    out = await graph_mod._docs_by_id(coll, [])
    assert out == {}


# ---------------------------------------------------------- _plasticity_overlay
async def test_plasticity_overlay_skips_edges_to_off_graph_nodes() -> None:
    """Line 135: continue when the dst is not in the surviving node set."""
    edges = [
        _FakeEdge(1, 2, hits=10, weight=1.0, bonus=0.5),  # both endpoints kept
        _FakeEdge(1, 9, hits=10, weight=1.0, bonus=0.5),  # dst 9 not in nodes
    ]
    coll = _FakeColl(plast_edges=edges)
    out = await graph_mod._plasticity_overlay(coll, nodes={1, 2}, habit_threshold=5)
    assert len(out) == 1
    assert out[0]["src"] == 1
    assert out[0]["dst"] == 2
    assert out[0]["is_habit"] is True
    assert out[0]["strength"] == pytest.approx(1.5)


# ---------------------------------------------------------- _attach_closure
def test_attach_closure_sets_zero_when_sr_is_none() -> None:
    """Lines 154-156: no SR → every edge gets m=0.0."""
    sr_edges = [{"src": 1, "dst": 2, "hits": 3}, {"src": 1, "dst": 4, "hits": 1}]
    graph_mod._attach_closure(None, sr_edges)
    assert all(e["m"] == 0.0 for e in sr_edges)


def test_attach_closure_pulls_m_from_sr_boost() -> None:
    """SR present → each edge gets m = sr.boost(src, [dst])[dst]."""

    class _SR:
        def boost(self, src: int, dsts: list[int]) -> dict[int, float]:
            return {d: 0.5 * d for d in dsts}

    sr_edges = [{"src": 1, "dst": 2, "hits": 3}, {"src": 1, "dst": 4, "hits": 1}]
    graph_mod._attach_closure(_SR(), sr_edges)
    assert sr_edges[0]["m"] == pytest.approx(1.0)
    assert sr_edges[1]["m"] == pytest.approx(2.0)


# ---------------------------------------------------------- _semantic_overlay
class _StubService:
    def __init__(
        self,
        semantic: _FakeColl,
        consolidate_full_at: int = 3,
    ) -> None:
        self.semantic = semantic

        class _Cfg:
            consolidate_confidence_full_at = consolidate_full_at

        self.config = _Cfg()


async def test_semantic_overlay_emits_nodes_and_edges_when_schema_ids_present() -> None:
    """Lines 192-215: non-empty schema_ids path. Schemas resolve via
    _docs_by_id and consolidation edges link source episodes to schemas."""
    schema_id = 42
    semantic = _FakeColl(
        documents={
            schema_id: (
                "schema text",
                {
                    "source_episode_ids": [1, 2],
                    "confidence": 0.6,
                    "user_asserted": False,
                },
            )
        }
    )
    service = _StubService(semantic=semantic)
    docs = {
        1: ("ep one", {"consolidated_into": schema_id}),
        2: ("ep two", {"consolidated_into": schema_id}),
        3: ("ep three", {"consolidated_into": 0}),  # not consolidated → no edge
    }
    nodes, edges = await graph_mod._semantic_overlay(
        service,  # type: ignore[arg-type]
        docs,
        semantic_ids={schema_id},
    )
    assert len(nodes) == 1
    node = nodes[0]
    assert node["id"] == schema_id
    assert node["key"] == f"sem:{schema_id}"
    assert node["kind"] == "semantic"
    assert node["source_count"] == 2
    assert node["confidence"] == pytest.approx(0.6)
    # Edges only for episodes pointing at schema_id.
    srcs = sorted(e["src"] for e in edges)
    assert srcs == [1, 2]
    assert all(e["dst"] == f"sem:{schema_id}" for e in edges)


async def test_semantic_overlay_empty_schema_ids_returns_empty() -> None:
    service = _StubService(semantic=_FakeColl())
    nodes, edges = await graph_mod._semantic_overlay(
        service,  # type: ignore[arg-type]
        docs={},
        semantic_ids=set(),
    )
    assert nodes == []
    assert edges == []


async def test_all_semantic_ids_enumerates_every_doc_up_to_limit() -> None:
    """Standalone (user-asserted) semantic memories with no consolidating
    episodes must still surface — otherwise they're invisible on the map."""
    semantic = _FakeColl(
        documents={
            1: ("orphan a", {"authored": True}),
            2: ("orphan b", {"authored": True}),
            3: ("schema c", {"source_episode_ids": [10, 11]}),
        }
    )
    ids, truncated = await graph_mod._all_semantic_ids(semantic, limit=10)
    assert ids == {1, 2, 3}
    assert truncated is False


async def test_all_semantic_ids_respects_limit_and_signals_truncation() -> None:
    semantic = _FakeColl(documents={i: (f"s{i}", {}) for i in range(5)})
    ids, truncated = await graph_mod._all_semantic_ids(semantic, limit=2)
    assert len(ids) == 2
    assert truncated is True


async def test_semantic_overlay_labels_orphans_as_memory_not_schema() -> None:
    """Authored (user-asserted) semantics get a ``memory #N`` label so
    they're not visually confused with consolidated schemas."""
    orphan_id, schema_id = 9, 7
    semantic = _FakeColl(
        documents={
            orphan_id: ("orphan", {"authored": True}),
            schema_id: ("schema", {"source_episode_ids": [1]}),
        }
    )
    service = _StubService(semantic=semantic, consolidate_full_at=4)
    nodes, _ = await graph_mod._semantic_overlay(
        service,  # type: ignore[arg-type]
        docs={},
        semantic_ids={orphan_id, schema_id},
    )
    by_id = {n["id"]: n for n in nodes}
    assert by_id[orphan_id]["label"] == f"memory #{orphan_id}"
    assert by_id[orphan_id]["user_asserted"] is True
    assert by_id[schema_id]["label"] == f"schema #{schema_id}"
    assert by_id[schema_id]["user_asserted"] is False


async def test_semantic_overlay_uses_default_confidence_when_metadata_missing() -> None:
    """Schema doc with no explicit ``confidence`` falls back to
    min(1, sources / full_at). Exercises the default arm in line 204."""
    schema_id = 7
    semantic = _FakeColl(documents={schema_id: ("s", {"source_episode_ids": [1]})})
    service = _StubService(semantic=semantic, consolidate_full_at=4)
    docs = {1: ("ep", {"consolidated_into": schema_id})}
    nodes, _ = await graph_mod._semantic_overlay(service, docs, {schema_id})  # type: ignore[arg-type]
    assert nodes[0]["confidence"] == pytest.approx(0.25)
