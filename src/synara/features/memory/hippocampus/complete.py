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
    q_{t+1} = normalize((1 - eta_0) q_0 + eta_0 x~_t)

The **inner completion score** at iterate ``q_t`` is the log-sum-exp:

    C(q_t) = (1/beta) * log sum_i exp(beta * <q_t, x_i>)

C is monotonically non-decreasing along the un-anchored (eta_0 = 1)
iteration and bounded above by ``max_i <q_T, x_i>``. The
eta_0-anchored variant trades strict monotonicity for robustness
against drift toward a spurious attractor far from the original
query — empirically more useful for under-specified queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..port import MemoryServicePort as MemoryService


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
    return v / max(n, 1e-12)


def completion_score(q: np.ndarray, X: np.ndarray, *, beta: float) -> float:
    """Compute inner completion score: (1/beta) * log sum_i exp(beta * <q, x_i>).

    Numerically stable by max-subtraction. Returns 0 for empty pattern set.
    """
    if X.shape[0] == 0:
        return 0.0
    sims = X @ q
    m = float(sims.max())
    return m + (1.0 / beta) * float(np.log(np.exp(beta * (sims - m)).sum()))


def attractor_step(
    q: np.ndarray,
    X: np.ndarray,
    *,
    beta: float,
    q0: np.ndarray,
    eta0: float,
) -> tuple[np.ndarray, float]:
    """One modern-Hopfield update + post-update completion score.

    q, q0: unit-normalised (D,). X: (N, D) of unit-normalised vectors.
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
    q_next = _normalize((1.0 - eta0) * q0 + eta0 * x_tilde)
    return q_next, score


async def _gather_candidates(
    service: MemoryService,
    q: list[float],
    *,
    ep_filter: dict[str, Any] | None,
    k: int,
) -> np.ndarray:
    """Search episodic + semantic stores; return stacked unit-norm embeddings.

    Returns empty (0,)-shape array if no hits."""
    rows: list[np.ndarray] = []
    for coll, flt in ((service.episodic, ep_filter), (service.semantic, None)):
        if await coll.count() == 0:
            continue
        try:
            hits = await coll.similarity_search(q, k=k, filter=flt)
        except (ValueError, RuntimeError):
            continue
        ids = [int(d.metadata.get("id", -1)) for d, _ in hits]
        ids = [i for i in ids if i >= 0]
        if not ids:
            continue
        emap = await coll.get_embeddings_by_ids(ids)
        for did in ids:
            v = emap.get(did)
            if v is None:
                continue
            rows.append(_normalize(np.asarray(v, dtype=np.float64)))
    if not rows:
        return np.zeros((0,), dtype=np.float64)
    return np.stack(rows, axis=0)


async def run(
    service: MemoryService,
    q0: list[float],
    *,
    ep_filter: dict[str, Any] | None,
    k_inner: int,
    iters: int,
    beta: float,
    eta0: float,
    eps: float = 1e-3,
) -> CompletionResult:
    """Iterate modern-Hopfield updates; return refined query + score trace.

    Stops early if score delta < eps or candidate set is empty.
    iters <= 0 returns q0 unchanged.
    """
    if iters <= 0:
        return CompletionResult(query=list(q0), scores=[], converged=True)

    q = _normalize(np.asarray(q0, dtype=np.float64))
    q0_arr = q.copy()
    scores: list[float] = []

    for _ in range(iters):
        X = await _gather_candidates(service, q.tolist(), ep_filter=ep_filter, k=k_inner)
        if X.shape[0] == 0:
            break
        q, score = attractor_step(q, X, beta=beta, q0=q0_arr, eta0=eta0)
        scores.append(score)
        if len(scores) >= 2 and abs(scores[-1] - scores[-2]) < eps:  # noqa: PLR2004
            return CompletionResult(query=q.tolist(), scores=scores, converged=True)

    return CompletionResult(query=q.tolist(), scores=scores, converged=False)
