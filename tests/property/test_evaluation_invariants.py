"""Property-based invariants for the evaluation kernels (Hypothesis).

Covers the invariants that matter for an honest, reproducible credit headline:

- metrics are invariant to row permutation (a model's AUC/Brier/log-loss/KS does
  not depend on row order - the analogue of the "prediction invariance to row
  permutation" property);
- ROC-AUC and PR-AUC stay within ``[0, 1]`` and Brier within ``[0, 1]``;
- the DeLong overfitting guard is sane: identical scores give a zero gap, and the
  Bonferroni ``n_comparisons`` correction monotonically inflates the p-value and
  is capped at 1.0 (the credit analogue of correcting for a trial grid - the
  recorded comparison count must be >= 1 and only ever makes the test stricter).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lendingclub_default._exceptions import ValidationError
from lendingclub_default.evaluation.delong import delong_auc_test
from lendingclub_default.evaluation.metrics import (
    brier_score,
    ks_statistic,
    log_loss,
    pr_auc,
    roc_auc,
)


def _labelled(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-class labels (both present) and noisy signal-bearing probabilities."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    # Guarantee at least one of each class.
    y[: max(1, n // 4)] = 1
    rng.shuffle(y)
    prob = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.1, size=n), 1e-4, 1 - 1e-4)
    return y, prob


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(40, 300))
def test_metrics_invariant_to_row_permutation(seed: int, n: int) -> None:
    """All headline metrics are unchanged by a permutation of the rows."""
    y, prob = _labelled(seed, n)
    perm = np.random.default_rng(seed + 1).permutation(n)
    yp, pp = y[perm], prob[perm]
    assert roc_auc(y, prob) == pytest.approx(roc_auc(yp, pp), abs=1e-12)
    assert pr_auc(y, prob) == pytest.approx(pr_auc(yp, pp), abs=1e-12)
    assert brier_score(y, prob) == pytest.approx(brier_score(yp, pp), abs=1e-12)
    assert log_loss(y, prob) == pytest.approx(log_loss(yp, pp), abs=1e-12)
    assert ks_statistic(y, prob) == pytest.approx(ks_statistic(yp, pp), abs=1e-12)


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(40, 300))
def test_ranking_metrics_in_unit_interval(seed: int, n: int) -> None:
    """ROC-AUC, PR-AUC, KS, and Brier all lie within their valid ranges."""
    y, prob = _labelled(seed, n)
    for value in (roc_auc(y, prob), pr_auc(y, prob), ks_statistic(y, prob), brier_score(y, prob)):
        assert 0.0 <= value <= 1.0


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(60, 400))
def test_delong_identical_scores_zero_gap(seed: int, n: int) -> None:
    """DeLong on identical scores yields a zero AUC gap and a p-value of 1.0."""
    y, prob = _labelled(seed, n)
    res = delong_auc_test(y, prob, prob)
    assert res.auc_diff == pytest.approx(0.0, abs=1e-10)
    assert res.p_value == pytest.approx(1.0, abs=1e-10)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(80, 400),
    k=st.integers(1, 25),
)
def test_delong_bonferroni_monotone_and_capped(seed: int, n: int, k: int) -> None:
    """The recorded comparison count >= 1 only ever inflates p (capped at 1.0).

    This is the credit overfitting guard: correcting for ``k`` comparisons (the
    analogue of a trial grid of size ``k``) makes the AUC-gap test stricter, never
    looser, and the corrected p stays a valid probability.
    """
    rng = np.random.default_rng(seed)
    y, prob = _labelled(seed, n)
    other = np.clip(prob + rng.normal(0, 0.2, size=n), 1e-4, 1 - 1e-4)
    single = delong_auc_test(y, prob, other, n_comparisons=1)
    multi = delong_auc_test(y, prob, other, n_comparisons=k)
    assert multi.n_comparisons == k
    assert multi.p_value >= single.p_value - 1e-12
    assert 0.0 <= multi.p_value <= 1.0
    assert multi.p_value == pytest.approx(min(1.0, single.p_value * k), rel=1e-9, abs=1e-12)


@pytest.mark.property
def test_delong_rejects_invalid_comparison_count() -> None:
    """A recorded comparison count below 1 is rejected (guard precondition)."""
    y, prob = _labelled(1, 100)
    other = prob.copy()
    with pytest.raises(ValidationError):
        delong_auc_test(y, prob, other, n_comparisons=0)
