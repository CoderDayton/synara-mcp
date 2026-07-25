"""CA3-style iterative pattern completion (modern Hopfield).

Implements the continuous, attention-flavoured generalisation of
Marr's (1971) CA3 autoassociative recall, following Ramsauer et al.
(2020). A query is iteratively refined by softmax-weighted
recombination of its near-neighbours; the iteration drives an inner
completion score (a log-sum-exp free-energy potential) toward a
fixed-point attractor of the stored-pattern landscape.

Math
----
Let ``X = [x_1, ..., x_N]^T`` be the matrix of stored unit-normalised
embeddings, ``beta > 0`` an inverse temperature, and ``eta_0 in (0, 1]``
an anchor strength toward the original query. One step:

    a_t   = X q_t                              # affinities (N,)
    w_t,i = softmax(beta * a_t)_i              # attention weights
    x~_t  = sum_i w_t,i * x_i                  # recombined memory
    q_{t+1} = normalize((1 - eta_0) q_0 + eta_0 (x~_t - c))

The **inner completion score** at iterate ``q_t`` is the log-sum-exp:

    C(q_t) = (1/beta) * log sum_i exp(beta * <q_t, x_i>)

C is monotonically non-decreasing along the un-anchored (eta_0 = 1)
iteration and bounded above by ``max_i <q_T, x_i>``. The
eta_0-anchored variant trades strict monotonicity for robustness
against drift toward a spurious attractor far from the original
query — empirically more useful for under-specified queries.

Task-prefix offset
------------------
The affinity ``X q_t`` is exactly the comparison a retrieval model is
trained for: ``q`` carries the query encoding, ``X`` the document one.
The *recombination* is not — ``x~_t`` is a convex combination of stored
document vectors, so blending it straight into a query-space anchor
produces an iterate in neither space. Under a model with asymmetric task
prefixes the two sit ~0.10 cosine apart, and the fixed point drifts off
both manifolds; the final search then runs with a hybrid vector, and the
relevance ceiling (calibrated on document vectors, see :mod:`.background`)
gates it on a scale it does not belong to.

``c = normalize(q_0^doc) - normalize(q_0^query)`` is that displacement,
measured on the query text itself by encoding it both ways. Subtracting
it translates the recombined memory into the query's frame before the
blend, which keeps every iterate in query space. ``c`` is the zero vector
for a symmetric model, which makes the whole correction vanish.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..port import MemoryServicePort as MemoryService

_log = logging.getLogger(__name__)
_NORM_FLOOR = 1e-12  # below this, treat as zero vector — avoids div-by-zero blow-ups


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Result of one CA3 completion run.

    query: refined unit-norm query (D,).
    scores: per-step log-sum-exp scores, in order.
    converged: True if score delta fell below eps.
    """

    query: list[float]
    scores: list[float]
    converged: bool


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < _NORM_FLOOR:
        return np.zeros_like(v)
    return v / n


def completion_score(q: np.ndarray, X: np.ndarray, *, beta: float) -> float:
    """Compute inner completion score: (1/beta) * log sum_i exp(beta * <q, x_i>).

    Numerically stable by max-subtraction. Returns 0 for empty pattern set.
    """
    if X.shape[0] == 0:
        return 0.0
    sims = X @ q
    m = float(sims.max())
    return m + (1.0 / beta) * float(np.log(np.exp(beta * (sims - m)).sum()))


def space_offset(q_query: np.ndarray, q_doc: Sequence[float] | None) -> np.ndarray | None:
    """Displacement from the query encoding to the document encoding.

    Both arguments are encodings of the *same* text, so their difference
    isolates what the task prefix did to it -- see the module docstring.
    ``None`` when there is nothing to correct: no document encoding, a
    width disagreement, or non-finite values. An all-zero result (a
    symmetric model) is returned as-is and is a harmless no-op.
    """
    if q_doc is None:
        return None
    doc = np.asarray(q_doc, dtype=np.float64)
    if doc.shape != q_query.shape or not np.all(np.isfinite(doc)):
        return None
    offset: np.ndarray = _normalize(doc) - _normalize(q_query)
    return offset


def attractor_step(
    q: np.ndarray,
    X: np.ndarray,
    *,
    beta: float,
    q0: np.ndarray,
    eta0: float,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """One modern-Hopfield update + post-update completion score.

    q, q0: unit-normalised (D,). X: (N, D) of unit-normalised vectors.
    offset: the query->document displacement from :func:`space_offset`,
    or ``None`` for a model that encodes both sides alike.
    Returns (q_next, score).
    """
    sims = X @ q
    m = float(sims.max())
    # Single softmax pass: reuse the shifted exp for both the
    # log-sum-exp score and the attention weights, and contract via
    # GEMV (w @ X) instead of broadcasting an (N, D) intermediate.
    exp_shift = np.exp(beta * (sims - m))
    total = float(exp_shift.sum())
    score = m + (1.0 / beta) * float(np.log(total))
    w = exp_shift / total
    x_tilde = w @ X
    if offset is not None:
        # ``x_tilde`` is a mixture of *document* vectors; ``q0`` is a
        # query vector. Translate the former into the latter's frame so
        # the blend below stays inside one space.
        x_tilde = x_tilde - offset
    q_next = _normalize((1.0 - eta0) * q0 + eta0 * x_tilde)
    return q_next, score


async def _gather_candidates(
    service: MemoryService,
    q: list[float],
    *,
    k: int,
) -> np.ndarray:
    """Search episodic + semantic stores; return stacked unit-norm embeddings.

    Returns empty (0,)-shape array if no hits."""
    rows: list[np.ndarray] = []
    for coll in (service.episodic, service.semantic):
        if await coll.count() == 0:
            continue
        try:
            hits = await coll.similarity_search(q, k=k)
            ids = [int(d.metadata.get("id", -1)) for d, _ in hits]
            ids = [i for i in ids if i >= 0]
            if not ids:
                continue
            emap = await coll.get_embeddings_by_ids(ids)
        except (ValueError, RuntimeError) as exc:
            # One leg's failure should not abort CA3 completion — the
            # other store may still anchor the iteration. But silent
            # swallow hides backend regressions; surface them at debug.
            _log.debug("CA3 candidate gather failed for one leg: %s", exc, exc_info=True)
            continue
        for did in ids:
            v = emap.get(did)
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                # Skip corrupt stored vectors instead of poisoning X.
                continue
            rows.append(_normalize(arr))
    if not rows:
        return np.zeros((0,), dtype=np.float64)
    return np.stack(rows, axis=0)


async def run(
    service: MemoryService,
    q0: list[float],
    *,
    k_inner: int,
    iters: int,
    beta: float,
    eta0: float,
    eps: float = 1e-3,
    q0_document: list[float] | None = None,
) -> CompletionResult:
    """Iterate modern-Hopfield updates; return refined query + score trace.

    ``q0_document`` is the same query text encoded the way stored
    documents are; supplying it keeps the iteration in query space under
    a model with asymmetric task prefixes (see the module docstring).
    Omitting it reproduces the uncorrected iteration exactly, which is
    the right behaviour for a symmetric embedder.

    Stops early if score delta < eps or candidate set is empty.
    iters <= 0 returns q0 unchanged.
    """
    if iters <= 0:
        return CompletionResult(query=list(q0), scores=[], converged=True)
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be a positive finite float")
    if not 0.0 < eta0 <= 1.0:
        raise ValueError("eta0 must be in (0, 1]")
    if eps < 0.0 or not np.isfinite(eps):
        raise ValueError("eps must be a non-negative finite float")
    if k_inner <= 0:
        raise ValueError("k_inner must be positive")

    q_in = np.asarray(q0, dtype=np.float64)
    if q_in.ndim != 1 or q_in.size == 0:
        raise ValueError("q0 must be a non-empty 1-D vector")
    if not np.all(np.isfinite(q_in)):
        raise ValueError("q0 must contain only finite values")
    q = _normalize(q_in)
    q0_arr = q.copy()
    offset = space_offset(q0_arr, q0_document)
    scores: list[float] = []

    for _ in range(iters):
        X = await _gather_candidates(service, q.tolist(), k=k_inner)
        if X.shape[0] == 0:
            break
        q_next, score = attractor_step(q, X, beta=beta, q0=q0_arr, eta0=eta0, offset=offset)
        # ``attractor_step`` normalises through ``_normalize``, which
        # returns an all-zero vector if the update collapsed (anchor and
        # retrieved pattern cancelled). A zero query is a degenerate
        # search vector (undefined cosine); abandon the refinement and
        # keep the last good query rather than propagating zeros into the
        # downstream similarity search.
        if float(np.linalg.norm(q_next)) < _NORM_FLOOR:
            break
        q = q_next
        scores.append(score)
        if len(scores) >= 2 and abs(scores[-1] - scores[-2]) < eps:  # noqa: PLR2004
            return CompletionResult(query=q.tolist(), scores=scores, converged=True)

    return CompletionResult(query=q.tolist(), scores=scores, converged=False)
