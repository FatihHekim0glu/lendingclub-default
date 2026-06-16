"""Parity oracles for the evaluation kernels vs. sklearn / scipy.

These assert that the hand-written kernels agree with the reference
implementations to ``1e-10`` (or to the documented two-sample-KS / DeLong
references), so the library can report the metrics without a runtime sklearn
dependency in hot paths. Also exercises the validation guards and the
``n_comparisons`` (Bonferroni) overfitting guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as sk_log_loss,
)

from lendingclub_default._exceptions import ValidationError
from lendingclub_default.evaluation.calibration import reliability_curve
from lendingclub_default.evaluation.delong import delong_auc_test
from lendingclub_default.evaluation.metrics import (
    brier_score,
    compute_metrics,
    ks_statistic,
    log_loss,
    pr_auc,
    roc_auc,
)
from lendingclub_default.evaluation.threshold import CostMatrix, threshold_sweep

TOL = 1e-10


def _scored(seed: int, n: int = 600, p: float = 0.18) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob): labels with rate ~p and a signal-bearing PD."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < p).astype(int)
    # Probabilities correlated with the label (so AUC is well above 0.5) but noisy.
    base = rng.uniform(0.02, 0.6, size=n)
    prob = np.clip(base + 0.25 * y + rng.normal(0, 0.05, size=n), 1e-4, 1 - 1e-4)
    return y, prob


# --------------------------------------------------------------------------- #
# metrics: vs sklearn.metrics / scipy to 1e-10                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_roc_auc_matches_sklearn(seed: int) -> None:
    """roc_auc agrees with sklearn.roc_auc_score (ties via mid-ranks)."""
    y, prob = _scored(seed)
    assert roc_auc(y, prob) == pytest.approx(roc_auc_score(y, prob), abs=TOL)


@pytest.mark.parity
def test_roc_auc_handles_ties() -> None:
    """roc_auc matches sklearn even with many exactly-tied scores."""
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.3).astype(int)
    score = rng.integers(0, 5, size=400).astype(float)  # heavy ties
    assert roc_auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=TOL)


@pytest.mark.parity
@pytest.mark.parametrize("seed", [0, 5, 99, 2024])
def test_pr_auc_matches_average_precision(seed: int) -> None:
    """pr_auc agrees with sklearn.average_precision_score (step-function AP)."""
    y, prob = _scored(seed)
    assert pr_auc(y, prob) == pytest.approx(average_precision_score(y, prob), abs=TOL)


@pytest.mark.parity
def test_brier_and_log_loss_match_sklearn() -> None:
    """brier_score / log_loss agree with sklearn to 1e-10."""
    y, prob = _scored(11)
    assert brier_score(y, prob) == pytest.approx(brier_score_loss(y, prob), abs=TOL)
    assert log_loss(y, prob) == pytest.approx(sk_log_loss(y, prob, labels=[0, 1]), abs=TOL)


@pytest.mark.parity
@pytest.mark.parametrize("seed", [0, 13, 808])
def test_ks_matches_scipy_two_sample(seed: int) -> None:
    """ks_statistic equals scipy.stats.ks_2samp between class score CDFs."""
    y, prob = _scored(seed)
    ref = stats.ks_2samp(prob[y == 1], prob[y == 0]).statistic
    assert ks_statistic(y, prob) == pytest.approx(float(ref), abs=TOL)


@pytest.mark.parity
def test_compute_metrics_bundle_consistent() -> None:
    """compute_metrics matches the individual kernels and reports base rate / n."""
    y, prob = _scored(21)
    bundle = compute_metrics(y, prob)
    assert bundle.roc_auc == pytest.approx(roc_auc(y, prob), abs=TOL)
    assert bundle.pr_auc == pytest.approx(pr_auc(y, prob), abs=TOL)
    assert bundle.brier == pytest.approx(brier_score(y, prob), abs=TOL)
    assert bundle.log_loss == pytest.approx(log_loss(y, prob), abs=TOL)
    assert bundle.ks == pytest.approx(ks_statistic(y, prob), abs=TOL)
    assert bundle.base_rate == pytest.approx(float(np.mean(y)), abs=TOL)
    assert bundle.n == y.shape[0]
    payload = bundle.to_dict()
    assert isinstance(payload["n"], int)
    assert isinstance(payload["roc_auc"], float)


@pytest.mark.parity
def test_metric_validation_guards() -> None:
    """Misaligned, single-class, or non-binary inputs raise ValidationError."""
    with pytest.raises(ValidationError):
        roc_auc(np.array([0, 1, 0]), np.array([0.1, 0.2]))  # length mismatch
    with pytest.raises(ValidationError):
        roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))  # single class
    with pytest.raises(ValidationError):
        ks_statistic(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))  # single class
    with pytest.raises(ValidationError):
        pr_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))  # no positive
    with pytest.raises(ValidationError):
        roc_auc(np.array([0, 2, 1]), np.array([0.1, 0.2, 0.3]))  # non-binary label


# --------------------------------------------------------------------------- #
# calibration: vs sklearn.calibration.calibration_curve / CalibratedClassifierCV
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_reliability_curve_matches_sklearn_calibration_curve(strategy: str) -> None:
    """reliability_curve points match sklearn.calibration_curve for both strategies."""
    y, prob = _scored(4, n=800)
    curve = reliability_curve(y, prob, n_bins=10, strategy=strategy)
    prob_true, prob_pred = calibration_curve(y, prob, n_bins=10, strategy=strategy)
    np.testing.assert_allclose(curve.mean_predicted, prob_pred, atol=TOL)
    np.testing.assert_allclose(curve.observed_frequency, prob_true, atol=TOL)
    # ECE is the count-weighted |pred - obs| and lies in [0, 1].
    assert 0.0 <= curve.expected_calibration_error <= 1.0
    assert sum(curve.bin_counts) == y.shape[0]


@pytest.mark.parity
def test_calibration_matches_calibrated_classifier_cv() -> None:
    """An isotonic CalibratedClassifierCV is well-calibrated under our ECE/Brier.

    The calibrated reliability curve should sit near the diagonal (low ECE) —
    the same notion CalibratedClassifierCV optimizes — and its Brier score should
    track the raw score's closely (calibrating an already-well-specified base
    model neither helps nor hurts much).
    """
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=(n, 3))
    logits = x @ np.array([1.2, -0.7, 0.5]) - 0.8
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    x_tr, x_te = x[: n // 2], x[n // 2 :]
    y_tr, y_te = y[: n // 2], y[n // 2 :]

    base = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
    raw = base.predict_proba(x_te)[:, 1]

    calibrated = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000), method="isotonic", cv=3
    ).fit(x_tr, y_tr)
    cal = calibrated.predict_proba(x_te)[:, 1]

    assert brier_score(y_te, cal) == pytest.approx(brier_score(y_te, raw), abs=0.01)
    curve = reliability_curve(y_te, cal, n_bins=10, strategy="quantile")
    assert curve.expected_calibration_error < 0.1


@pytest.mark.parity
def test_reliability_curve_guards() -> None:
    """Bad n_bins / strategy / out-of-range probabilities raise ValidationError."""
    y, prob = _scored(2)
    with pytest.raises(ValidationError):
        reliability_curve(y, prob, n_bins=0)
    with pytest.raises(ValidationError):
        reliability_curve(y, prob, strategy="nope")
    with pytest.raises(ValidationError):
        reliability_curve(np.array([0, 1]), np.array([0.5, 1.4]))  # prob > 1
    with pytest.raises(ValidationError):
        reliability_curve(np.array([0, 1, 0]), np.array([0.2, 0.8]))  # length mismatch


@pytest.mark.parity
def test_reliability_curve_to_dict_is_json_safe() -> None:
    """ReliabilityCurve.to_dict yields plain floats/ints for the API boundary."""
    y, prob = _scored(3, n=400)
    payload = reliability_curve(y, prob, n_bins=8).to_dict()
    assert isinstance(payload["expected_calibration_error"], float)
    assert all(isinstance(v, float) for v in payload["mean_predicted"])
    assert all(isinstance(v, int) for v in payload["bin_counts"])


# --------------------------------------------------------------------------- #
# DeLong: sane on identical vs different scores; Bonferroni guard               #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_delong_identical_scores_zero_difference() -> None:
    """Identical score vectors give zero AUC gap and a non-significant p-value."""
    y, prob = _scored(8)
    res = delong_auc_test(y, prob, prob)
    assert res.auc_a == pytest.approx(res.auc_b, abs=TOL)
    assert res.auc_diff == pytest.approx(0.0, abs=TOL)
    assert res.p_value == pytest.approx(1.0, abs=TOL)


@pytest.mark.parity
def test_delong_auc_values_match_roc_auc_score() -> None:
    """The AUCs DeLong reports match sklearn.roc_auc_score for each model."""
    y, prob = _scored(15, n=1000)
    rng = np.random.default_rng(15)
    worse = np.clip(prob + rng.normal(0, 0.25, size=prob.shape[0]), 1e-4, 1 - 1e-4)
    res = delong_auc_test(y, prob, worse)
    assert res.auc_a == pytest.approx(roc_auc_score(y, prob), abs=1e-8)
    assert res.auc_b == pytest.approx(roc_auc_score(y, worse), abs=1e-8)


@pytest.mark.parity
def test_delong_detects_real_difference() -> None:
    """A genuinely stronger model yields a positive, significant AUC gap."""
    y, prob = _scored(30, n=1500)
    rng = np.random.default_rng(31)
    # Strongly degrade model B by adding heavy noise -> lower AUC.
    worse = np.clip(prob + rng.normal(0, 0.5, size=prob.shape[0]), 1e-4, 1 - 1e-4)
    res = delong_auc_test(y, prob, worse)
    assert res.auc_diff > 0.0
    assert res.std_error > 0.0
    assert res.p_value < 0.05


@pytest.mark.parity
def test_delong_bonferroni_scales_pvalue() -> None:
    """n_comparisons > 1 Bonferroni-inflates the p-value (capped at 1.0)."""
    y, prob = _scored(40, n=1200)
    rng = np.random.default_rng(41)
    worse = np.clip(prob + rng.normal(0, 0.3, size=prob.shape[0]), 1e-4, 1 - 1e-4)
    single = delong_auc_test(y, prob, worse, n_comparisons=1)
    multi = delong_auc_test(y, prob, worse, n_comparisons=10)
    assert multi.n_comparisons == 10
    assert multi.p_value == pytest.approx(min(1.0, single.p_value * 10), rel=1e-9)
    assert multi.p_value >= single.p_value


@pytest.mark.parity
def test_delong_guards() -> None:
    """Misaligned lengths, single class, non-binary, n_comparisons < 1 all raise."""
    y, prob = _scored(50)
    with pytest.raises(ValidationError):
        delong_auc_test(y, prob, prob[:-1])  # length mismatch
    with pytest.raises(ValidationError):
        delong_auc_test(np.ones_like(y), prob, prob)  # single class
    with pytest.raises(ValidationError):
        delong_auc_test(2 * np.ones_like(y), prob, prob)  # non-binary labels
    with pytest.raises(ValidationError):
        delong_auc_test(y, prob, prob, n_comparisons=0)


@pytest.mark.parity
def test_delong_to_dict_is_json_safe() -> None:
    """DeLongResult.to_dict yields plain floats and an int comparison count."""
    y, prob = _scored(55, n=500)
    rng = np.random.default_rng(55)
    other = np.clip(prob + rng.normal(0, 0.2, size=prob.shape[0]), 1e-4, 1 - 1e-4)
    payload = delong_auc_test(y, prob, other, n_comparisons=3).to_dict()
    assert isinstance(payload["n_comparisons"], int)
    assert payload["n_comparisons"] == 3
    assert all(
        isinstance(payload[k], float) for k in ("auc_a", "auc_b", "auc_diff", "z", "p_value")
    )


# --------------------------------------------------------------------------- #
# threshold sweep                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_threshold_sweep_finds_cost_minimizer() -> None:
    """threshold_sweep returns aligned arrays and a cost-minimizing threshold."""
    y, prob = _scored(60, n=900)
    sweep = threshold_sweep(y, prob, n_thresholds=101)
    assert len(sweep.thresholds) == len(sweep.expected_cost) == 101
    assert sweep.best_cost == pytest.approx(min(sweep.expected_cost), abs=TOL)
    idx = sweep.thresholds.index(sweep.best_threshold)
    assert sweep.expected_cost[idx] == pytest.approx(sweep.best_cost, abs=TOL)
    assert sweep.base_rate == pytest.approx(float(np.mean(y)), abs=TOL)


@pytest.mark.parity
def test_threshold_sweep_brute_force_matches() -> None:
    """The reported best cost matches an independent brute-force confusion count."""
    y, prob = _scored(61, n=500)
    cm = CostMatrix(cost_fn=1.0, cost_fp=0.25, cost_tp=-0.1, cost_tn=0.0)
    sweep = threshold_sweep(y, prob, cost_matrix=cm, n_thresholds=51)
    thresholds = np.linspace(0.0, 1.0, 51)
    n = y.shape[0]
    brute = []
    for t in thresholds:
        pred = prob >= t
        tp = np.count_nonzero(pred & (y == 1))
        fp = np.count_nonzero(pred & (y == 0))
        fn = np.count_nonzero(~pred & (y == 1))
        tn = np.count_nonzero(~pred & (y == 0))
        brute.append((cm.cost_tp * tp + cm.cost_fp * fp + cm.cost_fn * fn + cm.cost_tn * tn) / n)
    np.testing.assert_allclose(sweep.expected_cost, brute, atol=TOL)


@pytest.mark.parity
def test_threshold_sweep_guards() -> None:
    """Length mismatch and n_thresholds < 2 raise ValidationError."""
    y, prob = _scored(70)
    with pytest.raises(ValidationError):
        threshold_sweep(y, prob[:-1])
    with pytest.raises(ValidationError):
        threshold_sweep(y, prob, n_thresholds=1)
