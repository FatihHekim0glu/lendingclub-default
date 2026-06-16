"""Plotly figure-builder tests (plots.py group).

Asserts each builder returns a Plotly figure that serializes to a valid, finite
``{data, layout}`` blob (the API/frontend contract), that the ROC/PR step points
match sklearn's curves, that the calibration figure consumes a real
:class:`ReliabilityCurve`, and that the validation guards fire. Also pins import
purity: importing ``plots`` must not import Plotly.
"""

from __future__ import annotations

import importlib
import math
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from lendingclub_default._exceptions import ValidationError
from lendingclub_default.evaluation.calibration import reliability_curve
from lendingclub_default.plots import (
    calibration_figure,
    figure_to_dict,
    pr_figure,
    roc_figure,
    score_distribution_figure,
)

_SEED = 20260616


def _scored(n: int = 800, *, seed: int = _SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return a binary label vector and a correlated probability vector in [0, 1]."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.18).astype(int)
    score = np.clip(0.18 + 0.30 * y + rng.normal(0.0, 0.18, size=n), 0.0, 1.0)
    return y, score


def _assert_valid_figure_dict(payload: dict[str, object]) -> None:
    """A serialized figure must be a {data, layout} dict with finite trace coords."""
    assert isinstance(payload, dict)
    assert "data" in payload and "layout" in payload
    assert isinstance(payload["data"], list) and len(payload["data"]) >= 1
    assert isinstance(payload["layout"], dict)
    for trace in payload["data"]:
        for axis in ("x", "y"):
            values = trace.get(axis)
            if values is None:
                continue
            for v in values:
                if v is None:
                    continue
                assert math.isfinite(float(v)), f"non-finite {axis} value {v!r}"


@pytest.mark.unit
def test_roc_figure_is_valid_and_finite() -> None:
    """roc_figure serializes to a finite {data, layout} blob with two traces."""
    y, score = _scored()
    payload = figure_to_dict(roc_figure(y, score))
    _assert_valid_figure_dict(payload)
    # ROC plus the no-skill diagonal.
    assert len(payload["data"]) == 2
    assert "ROC" in payload["layout"]["title"]["text"]


@pytest.mark.unit
def test_pr_figure_is_valid_and_finite() -> None:
    """pr_figure serializes to a finite {data, layout} blob with a base-rate line."""
    y, score = _scored()
    payload = figure_to_dict(pr_figure(y, score))
    _assert_valid_figure_dict(payload)
    assert len(payload["data"]) == 2


@pytest.mark.unit
def test_score_distribution_figure_is_valid_and_finite() -> None:
    """score_distribution_figure overlays two finite histograms."""
    y, prob = _scored()
    payload = figure_to_dict(score_distribution_figure(y, prob, n_bins=30))
    _assert_valid_figure_dict(payload)
    assert payload["layout"]["barmode"] == "overlay"
    assert len(payload["data"]) == 2


@pytest.mark.unit
def test_calibration_figure_from_reliability_curve() -> None:
    """calibration_figure consumes a real ReliabilityCurve and stays finite."""
    y, prob = _scored()
    curve = reliability_curve(y, prob, n_bins=8)
    payload = figure_to_dict(calibration_figure(curve))
    _assert_valid_figure_dict(payload)
    # Perfect-calibration diagonal plus the model curve.
    assert len(payload["data"]) == 2
    assert "ECE" in payload["layout"]["title"]["text"]


@pytest.mark.unit
def test_roc_curve_area_matches_sklearn_auc() -> None:
    """The plotted ROC trace traces the same curve as sklearn (area == AUC).

    The trapezoidal area under the figure's (fpr, tpr) points equals
    ``sklearn.metrics.roc_auc_score``, the title carries that AUC, and the curve
    is monotone non-decreasing in both axes (a valid ROC).
    """
    y, score = _scored()
    payload = figure_to_dict(roc_figure(y, score))
    roc_trace = payload["data"][0]
    fpr = np.asarray(roc_trace["x"], dtype="float64")
    tpr = np.asarray(roc_trace["y"], dtype="float64")
    auc_ref = float(roc_auc_score(y, score))
    np.testing.assert_allclose(np.trapezoid(tpr, fpr), auc_ref, atol=1e-10)
    assert np.all(np.diff(fpr) >= -1e-12)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert f"{auc_ref:.3f}" in payload["layout"]["title"]["text"]


@pytest.mark.unit
def test_pr_curve_area_matches_sklearn_average_precision() -> None:
    """The plotted PR trace yields the same average precision as sklearn.

    The step-sum ``sum_k (R_k - R_{k-1}) * P_k`` over the figure's (recall,
    precision) points equals ``sklearn.metrics.average_precision_score``.
    """
    y, score = _scored()
    payload = figure_to_dict(pr_figure(y, score))
    pr_trace = payload["data"][0]
    recall = np.asarray(pr_trace["x"], dtype="float64")
    precision = np.asarray(pr_trace["y"], dtype="float64")
    # Drop the appended (recall=0, precision=1) endpoint, then prepend the origin
    # so the first delta-recall is recall[0] (the average-precision convention).
    recall = np.r_[0.0, recall[:-1]]
    precision = precision[:-1]
    ap_curve = float(np.sum(np.diff(recall) * precision))
    ap_ref = float(average_precision_score(y, score))
    np.testing.assert_allclose(ap_curve, ap_ref, atol=1e-10)


@pytest.mark.unit
def test_roc_figure_rejects_single_class() -> None:
    """A single-class label vector cannot define an ROC curve."""
    score = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValidationError):
        roc_figure(np.zeros(50, dtype=int), score)


@pytest.mark.unit
def test_figures_reject_misaligned_inputs() -> None:
    """Length-mismatched inputs raise ValidationError across the builders."""
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValidationError):
        roc_figure(y, score)
    with pytest.raises(ValidationError):
        pr_figure(y, score)


@pytest.mark.unit
def test_score_distribution_rejects_out_of_range_and_bad_bins() -> None:
    """Probabilities must be in [0, 1] and the bin count positive."""
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValidationError):
        score_distribution_figure(y, np.array([0.1, 1.5, 0.2, 0.3]))
    with pytest.raises(ValidationError):
        score_distribution_figure(y, np.array([0.1, 0.2, 0.2, 0.3]), n_bins=0)


@pytest.mark.unit
def test_roc_figure_rejects_empty_inputs() -> None:
    """Empty input vectors raise ValidationError."""
    empty = np.array([], dtype="float64")
    with pytest.raises(ValidationError):
        roc_figure(empty, empty)


@pytest.mark.unit
def test_roc_figure_rejects_non_binary_labels() -> None:
    """Non-{0, 1} labels raise ValidationError."""
    y = np.array([0, 2, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValidationError):
        roc_figure(y, score)


@pytest.mark.unit
def test_calibration_figure_rejects_empty_curve() -> None:
    """A reliability curve with no bins cannot be plotted."""
    from lendingclub_default.evaluation.calibration import ReliabilityCurve

    empty = ReliabilityCurve(
        mean_predicted=[],
        observed_frequency=[],
        bin_counts=[],
        expected_calibration_error=0.0,
        n_bins=0,
    )
    with pytest.raises(ValidationError):
        calibration_figure(empty)


@pytest.mark.unit
def test_calibration_figure_rejects_non_finite_points() -> None:
    """A reliability curve carrying NaN/inf points cannot be plotted."""
    from lendingclub_default.evaluation.calibration import ReliabilityCurve

    bad = ReliabilityCurve(
        mean_predicted=[0.1, float("nan")],
        observed_frequency=[0.1, 0.2],
        bin_counts=[10, 10],
        expected_calibration_error=0.05,
        n_bins=2,
    )
    with pytest.raises(ValidationError):
        calibration_figure(bad)


@pytest.mark.unit
def test_plots_module_import_is_plotly_free() -> None:
    """Importing lendingclub_default.plots must not pull Plotly onto sys.path.

    Runs in a fresh interpreter so the assertion is not polluted by other tests
    that have already imported Plotly via figure builders.
    """
    code = (
        "import sys; import lendingclub_default.plots; "
        "assert 'plotly' not in sys.modules; print('pure')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pure" in result.stdout


@pytest.mark.unit
def test_plots_module_reimports_cleanly() -> None:
    """The module can be re-imported without side effects (sanity)."""
    module = importlib.import_module("lendingclub_default.plots")
    assert hasattr(module, "roc_figure")
    assert hasattr(module, "figure_to_dict")
