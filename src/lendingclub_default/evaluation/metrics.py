"""Ranking and probabilistic evaluation metrics for the default classifier.

The honest headline is ROC-AUC / PR-AUC / Brier — NEVER accuracy, NEVER
profit/ROI. These functions are validated to ``1e-10`` against ``sklearn.metrics``
in the parity suite, so the library can report them without a runtime sklearn
dependency in hot paths.

- :func:`roc_auc` — area under the ROC curve (ranking quality).
- :func:`pr_auc` — average precision / area under the PR curve (the metric that
  matters under a ~15% positive base rate).
- :func:`brier_score` — mean squared error of the calibrated PD (calibration).
- :func:`log_loss` — cross-entropy of the calibrated PD.
- :func:`ks_statistic` — Kolmogorov-Smirnov separation of the two score CDFs.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# quantcore-candidate: new code (credit metrics); parity oracle = sklearn.metrics
# (1e-10) + scipy KS.


@dataclass(frozen=True, slots=True)
class MetricBundle:
    """Immutable bundle of the headline evaluation metrics.

    Attributes
    ----------
    roc_auc:
        Area under the ROC curve.
    pr_auc:
        Average precision (area under the precision-recall curve).
    brier:
        Brier score (lower is better).
    log_loss:
        Mean cross-entropy (lower is better).
    ks:
        Kolmogorov-Smirnov statistic between default/non-default score CDFs.
    base_rate:
        The positive (default) base rate the metrics were computed against.
    n:
        Number of scored observations.
    """

    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    ks: float
    base_rate: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the bundle."""
        out = asdict(self)
        return {k: (float(v) if k != "n" else int(v)) for k, v in out.items()}


def roc_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Area under the ROC curve (the Mann-Whitney U statistic, normalized).

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores or probabilities (higher = riskier).

    Returns
    -------
    float
        ROC-AUC in ``[0, 1]``.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("roc_auc is not yet implemented.")


def pr_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Average precision (area under the precision-recall curve).

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores or probabilities.

    Returns
    -------
    float
        Average precision in ``[0, 1]``.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("pr_auc is not yet implemented.")


def brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Brier score: mean squared error of the predicted probabilities.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.

    Returns
    -------
    float
        The Brier score (lower is better).

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("brier_score is not yet implemented.")


def log_loss(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Mean binary cross-entropy (log loss) of the predicted probabilities.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities; clipped by ``EPS`` away from ``{0, 1}``.

    Returns
    -------
    float
        The mean log loss (lower is better).

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("log_loss is not yet implemented.")


def ks_statistic(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Kolmogorov-Smirnov separation between default and non-default score CDFs.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores.

    Returns
    -------
    float
        The KS statistic in ``[0, 1]`` (max vertical gap between the two CDFs).

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("ks_statistic is not yet implemented.")


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> MetricBundle:
    """Compute the full headline metric bundle in one call.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.

    Returns
    -------
    MetricBundle
        ROC-AUC, PR-AUC, Brier, log-loss, KS, base rate, and ``n``.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("compute_metrics is not yet implemented.")
