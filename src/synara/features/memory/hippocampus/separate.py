"""DG-style pattern separation: random expansive projection + k-WTA.

Marr (1971) and McNaughton & Morris (1987): the dentate gyrus expands
each entorhinal-cortex input into a much larger granule-cell layer
(~10x in the rat) and a competitive k-winners-take-all mechanism
keeps only the top ~5% active. Similar inputs end up with
*more orthogonal* sparse codes than their dense embeddings — the
defining computation of pattern separation.

Math
----
Given a unit-normalised embedding ``x in R^D``, build a random
projection ``W in R^{M x D}`` with i.i.d. Gaussian entries scaled by
``1 / sqrt(D)`` (Johnson-Lindenstrauss compatible), where ``M = e D``.
Then

    Phi(x) = TopK_k( ReLU(W x) ),    k = alpha * M

with ``alpha`` the target sparsity (~0.05). The *support* of Phi(x) —
the set of active indices — is what we compare. Pattern-separation
strength is measured by Jaccard overlap of supports:

    J(Phi(x), Phi(y)) = |A intersect B| / |A union B|,
    A = supp Phi(x),  B = supp Phi(y)

For two embeddings with cosine similarity ``1 - eps``, the expected
support overlap shrinks roughly like ``(1 - eps)^c`` with
``c = log(M / k)``. Small input differences become large
representation differences — exactly the pattern-separation property.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


class DGProjector:
    """Random expansive projection + k-WTA sparsification.

    Seeded so same config produces same supports across restarts.
    """

    def __init__(
        self,
        dim: int,
        *,
        expansion: int = 4,
        sparsity: float = 0.05,
        seed: int = 0,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if expansion < 1:
            raise ValueError("expansion must be >= 1")
        if not 0.0 < sparsity < 1.0:
            raise ValueError("sparsity must be in (0, 1)")
        self.dim = dim
        self.expansion = expansion
        self.sparsity = sparsity
        self.M = int(expansion * dim)
        self.k = max(1, int(sparsity * self.M))
        rng = np.random.default_rng(seed)
        # 1/sqrt(D) scale keeps Var((Wx)_j) ~ ||x||^2 / D bounded.
        self.W = (rng.standard_normal((self.M, dim)) / np.sqrt(dim)).astype(np.float32)

    def support(self, x: Sequence[float]) -> tuple[int, ...]:
        """Return sorted tuple of active indices (up to k).

        Hashable and JSON-serialisable as list for document metadata."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 1 or x_arr.shape[0] != self.dim:
            raise ValueError(f"expected vector of dim {self.dim}, got shape {x_arr.shape}")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("input vector must contain only finite values")
        h = self.W @ x_arr
        relu = np.maximum(h, 0.0)
        nonzero = int((relu > 0).sum())
        if nonzero == 0:
            return ()
        if nonzero <= self.k:
            # ``np.flatnonzero(...).tolist()`` already yields native
            # Python ints (numpy int64 -> int on tolist), so the prior
            # ``int(i)`` generator was a redundant per-element cast.
            return tuple(sorted(np.flatnonzero(relu > 0).tolist()))
        # ``argpartition`` finds the top-k unsorted in O(M).
        idx = np.argpartition(-relu, self.k - 1)[: self.k]
        return tuple(sorted(idx.tolist()))


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard overlap of two index sets.

    1.0 for two empty sets (equal); 0.0 if one is empty and other is not."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return len(sa & sb) / union
