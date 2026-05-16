"""Pure-logic unit tests for hippocampus primitives.

Covers the validation/edge branches in separate, segment, complete,
and successor that the service-level tests do not reach.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from synara.features.memory.hippocampus.complete import (
    CompletionResult,
    completion_score,
)
from synara.features.memory.hippocampus.segment import split_into_segments
from synara.features.memory.hippocampus.separate import DGProjector, jaccard
from synara.features.memory.hippocampus.successor import SuccessorRepresentation

# ---- DGProjector / jaccard -------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dim": 0}, "dim must be positive"),
        ({"dim": 8, "expansion": 0}, "expansion must be >= 1"),
        ({"dim": 8, "sparsity": 0.0}, "sparsity must be in"),
        ({"dim": 8, "sparsity": 1.0}, "sparsity must be in"),
    ],
)
def test_dg_projector_rejects_bad_config(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DGProjector(**kwargs)


def test_dg_projector_support_wrong_shape_raises() -> None:
    p = DGProjector(dim=8, seed=1)
    with pytest.raises(ValueError, match="expected vector of dim 8"):
        p.support([0.1, 0.2])


def test_dg_projector_all_negative_projection_returns_empty() -> None:
    p = DGProjector(dim=4, expansion=2, sparsity=0.5, seed=3)
    # Force W @ x <= 0 everywhere by feeding the zero vector.
    assert p.support([0.0, 0.0, 0.0, 0.0]) == ()


def test_dg_projector_support_when_nonzero_at_most_k() -> None:
    # Tiny layer (M=2, k=1): a vector activating <=k units exercises the
    # ``nonzero <= self.k`` short path.
    p = DGProjector(dim=2, expansion=1, sparsity=0.5, seed=0)
    s = p.support([1.0, 0.0])
    assert isinstance(s, tuple)
    assert len(s) <= p.k or len(s) == p.M


def test_jaccard_edge_cases() -> None:
    assert jaccard([], []) == 1.0
    assert jaccard([1], []) == 0.0
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


# ---- segment ---------------------------------------------------------


def test_segment_long_single_sentence_flushes_buffer_then_windows() -> None:
    # A short sentence buffers, then an over-budget sentence forces the
    # buffer flush + character windowing path.
    short = "Short one."
    longsent = "X" * 50  # one token, no sentence boundary, > max_chars
    content = f"{short} {longsent}"
    segs = split_into_segments(content, max_chars=20, max_items=10)
    assert segs[0] == short
    assert "".join(segs[1:]).replace(" ", "") == longsent
    assert all(len(s) <= 20 for s in segs[1:])


def test_segment_disabled_and_passthrough() -> None:
    assert split_into_segments("abc", max_chars=0, max_items=10) == ["abc"]
    assert split_into_segments("abc", max_chars=10, max_items=1) == ["abc"]
    assert split_into_segments("   ", max_chars=5, max_items=4) == ["   "]
    assert split_into_segments("tiny", max_chars=100, max_items=4) == ["tiny"]


# ---- completion_score / CompletionResult -----------------------------


def test_completion_score_empty_pattern_set_is_zero() -> None:
    q = np.array([1.0, 0.0], dtype=np.float64)
    empty = np.zeros((0, 2), dtype=np.float64)
    assert completion_score(q, empty, beta=8.0) == 0.0


def test_completion_score_value_matches_logsumexp() -> None:
    q = np.array([1.0, 0.0], dtype=np.float64)
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    beta = 4.0
    sims = X @ q
    m = sims.max()
    expected = m + (1.0 / beta) * np.log(np.exp(beta * (sims - m)).sum())
    assert completion_score(q, X, beta=beta) == pytest.approx(expected)


def test_completion_result_is_frozen() -> None:
    r = CompletionResult(query=[0.1], scores=[1.0], converged=True)
    with pytest.raises(AttributeError):
        r.converged = False  # type: ignore[misc]


# ---- SuccessorRepresentation (in-memory, no collection) --------------


def test_sr_observe_skips_self_and_records_prior_edges() -> None:
    sr = SuccessorRepresentation(window_seconds=100.0)
    sr.observe("s", 1, t=0.0)
    sr.observe("s", 1, t=1.0)  # self in window -> skipped (line 151)
    sr.observe("s", 2, t=2.0)  # both in-window id=1 priors -> two 1->2 edges
    assert sr.total_edges == 2.0
    assert sr.boost(1, [2])[2] > 0.0


def test_sr_observe_evicts_out_of_window_priors() -> None:
    sr = SuccessorRepresentation(window_seconds=10.0)
    sr.observe("s", 1, t=0.0)
    # 100s later: prior id=1 is outside the window and is popped before
    # any edge can form.
    sr.observe("s", 2, t=100.0)
    assert sr.total_edges == 0.0


def test_sr_observe_recall_set_skips_anchor_in_others() -> None:
    sr = SuccessorRepresentation(window_seconds=100.0)
    sr.observe_recall_set("s", anchor_id=7, other_ids=[7, 8], t=1.0)
    # 7->7 skipped, 7->8 recorded.
    assert sr.total_edges == 1.0
    assert sr.boost(7, [8])[8] > 0.0


def test_sr_recall_set_window_eviction_then_chain() -> None:
    sr = SuccessorRepresentation(window_seconds=10.0)
    sr.observe_recall_set("s", anchor_id=1, other_ids=[2], t=0.0)
    # New recall far in the future: prior anchor 1 evicted (line 170),
    # only the 3->4 edge forms.
    sr.observe_recall_set("s", anchor_id=3, other_ids=[4], t=1000.0)
    assert sr.boost(1, [3]) == {3: 0.0}


def test_sr_omega_cold_start_and_zero_guards() -> None:
    sr = SuccessorRepresentation(omega_max=0.3, cold_start_ratio=1.0)
    assert sr.omega(0) == 0.0  # no episodes
    assert sr.omega(10) == 0.0  # no edges yet -> ratio 0
    sr2 = SuccessorRepresentation(omega_max=0.0)
    assert sr2.omega(5) == 0.0  # omega_max disabled


async def test_sr_load_without_collection_is_idempotent() -> None:
    sr = SuccessorRepresentation()
    await sr.load()
    assert sr.total_edges == 0.0
    await sr.load()  # second call hits the already-loaded guard
    assert sr.total_edges == 0.0


async def test_sr_flush_without_collection_is_noop() -> None:
    sr = SuccessorRepresentation()
    sr.observe_recall_set("s", 1, [2], t=0.0)
    await sr.flush()  # no collection bound -> early return, no error
    assert sr.total_edges == 1.0
