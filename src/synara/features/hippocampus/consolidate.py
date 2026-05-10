"""Episodic -> semantic consolidation.

Clusters unconsolidated episodes (optionally restricted to one session)
and writes a semantic schema per cluster. Each schema's text is a
heuristic gist (no LLM) — sufficient to expose cluster content to
downstream recall, deliberately replaceable with a real summariser.

Source episodes are marked ``consolidated_into=<schema_id>`` so the
``forget`` pass knows their gist is preserved upstream and can prune
them more aggressively than uncosolidated ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .service import UNCONSOLIDATED, now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import HippocampusService


async def run(
    service: HippocampusService,
    *,
    session_id: str | None = None,
    n_clusters: int | None = None,
    min_cluster_size: int | None = None,
) -> list[dict[str, Any]]:
    flt: dict[str, Any] = {"consolidated_into": UNCONSOLIDATED}
    if session_id:
        flt["session_id"] = session_id

    candidates = await service.episodic.get_documents(flt)
    floor = min_cluster_size or service.config.consolidate_min_cluster
    if len(candidates) < floor:
        return []

    n = n_clusters or max(1, int(math.sqrt(len(candidates))))
    n = max(1, min(n, len(candidates)))
    try:
        result = await service.episodic.cluster(
            n_clusters=n,
            algorithm="minibatch_kmeans",
            filter=flt,
            random_state=0,
        )
    except (ValueError, ImportError):
        # Too few points for the requested algorithm or sklearn missing.
        return []

    groups: dict[int, list[int]] = {}
    for label, ep_id in zip(result.labels, result.doc_ids, strict=True):
        groups.setdefault(int(label), []).append(int(ep_id))

    ep_lookup: dict[int, tuple[str, dict[str, Any]]] = {
        int(ep_id): (text, dict(md)) for ep_id, text, md in candidates
    }
    formed: list[dict[str, Any]] = []
    total = max(1, len(candidates))

    for ep_ids in groups.values():
        members = [ep_lookup[eid] for eid in ep_ids if eid in ep_lookup]
        if len(members) < floor:
            continue

        head_text, _head_md = max(members, key=lambda m: float(m[1].get("salience", 0.0)))
        tag_union = sorted(
            {t for _, md in members for t in (md.get("tags") or []) if isinstance(t, str)}
        )
        summary = build_gist(head_text, [m[0] for m in members])
        confidence = min(1.0, len(ep_ids) / total)
        now = now_seconds()
        sem_meta: dict[str, Any] = {
            "source_episode_ids": list(ep_ids),
            "tags": tag_union,
            "confidence": float(confidence),
            "created_at": now,
            "updated_at": now,
        }
        sem_ids = await service.semantic.add_texts(
            [summary],
            metadatas=[sem_meta],
            embeddings=await service.vectorise([summary]),
        )
        sem_id = int(sem_ids[0])
        await service.semantic.update_metadata([(sem_id, {"id": sem_id})])

        await service.episodic.update_metadata(
            [(eid, {"consolidated_into": sem_id}) for eid in ep_ids]
        )
        formed.append(
            {
                "id": sem_id,
                "summary": summary,
                "source_episode_ids": list(ep_ids),
                "confidence": float(confidence),
                "tags": tag_union,
            }
        )
    return formed


def build_gist(headline: str, texts: Sequence[str]) -> str:
    """Heuristic semantic gist (no LLM).

    First non-empty line of the highest-salience episode becomes the
    headline; deduplicated leads from sibling episodes follow as bullets.
    """

    def _clip(t: str) -> str:
        line = t.strip().splitlines()[0] if t.strip() else ""
        return line[:200]

    head = _clip(headline)
    leads = sorted({_clip(t) for t in texts if t.strip()} - {head})
    if not leads:
        return head
    bullets = "\n".join(f"- {line}" for line in leads if line)
    return f"{head}\n\nRelated:\n{bullets}" if bullets else head
