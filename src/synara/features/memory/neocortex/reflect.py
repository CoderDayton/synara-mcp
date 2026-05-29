"""Self-reflection pass: relevant semantic schemas + recent episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..port import MemoryServicePort as MemoryService


async def run(
    service: MemoryService,
    *,
    session_id: str,
    query: str | None = None,
    k: int = 5,
) -> dict[str, Any]:
    if not session_id:
        raise ValidationError("session_id must be non-empty")
    if k <= 0:
        raise ValidationError("k must be positive")
    cfg = service.config
    if cfg.max_session_id_chars and len(session_id) > cfg.max_session_id_chars:
        raise ValidationError(
            f"session_id exceeds max_session_id_chars ({cfg.max_session_id_chars})"
        )
    if cfg.max_recall_k and k > cfg.max_recall_k:
        raise ValidationError(f"k exceeds max_recall_k ({cfg.max_recall_k})")

    episodes = await service.episodic.get_documents(
        {"session_id": session_id}, limit=max(50, k * 5)
    )
    recent = sorted(
        episodes,
        key=lambda r: float(r[2].get("last_accessed", 0.0)),
        reverse=True,
    )[:k]

    seed: str | None = query
    if seed is None:
        for _, _, md in recent:
            tags = md.get("tags") or []
            if tags:
                seed = str(tags[0])
                break

    sem_results: list[dict[str, Any]] = []
    if seed and await service.semantic.count() > 0:
        # Reconcile the HNSW index before searching: between an encode and
        # the first consolidate the index can be empty even though
        # ``count() > 0``, so a direct ``similarity_search`` would return
        # nothing. Mirrors the guard ``recall`` runs at its top.
        await service._ensure_index_ready()
        seed_q = await service.query_arg(seed)
        for doc, dist in await service.semantic.similarity_search(seed_q, k=k):
            sem_results.append(
                {
                    "id": int(doc.metadata.get("id", -1)),
                    "summary": doc.page_content,
                    "distance": float(dist),
                    "tags": list(doc.metadata.get("tags") or []),
                }
            )

    return {
        "session_id": session_id,
        "schemas": sem_results,
        "recent_episodes": [
            {"id": int(ep_id), "content": text, "metadata": dict(md)} for ep_id, text, md in recent
        ],
    }
