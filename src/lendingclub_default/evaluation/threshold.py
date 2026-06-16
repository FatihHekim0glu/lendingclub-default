"""Cost-matrix threshold sweep (reported, NEVER baked into the headline).

A calibrated PD ranks risk; turning it into an accept/reject decision needs a
threshold, which depends on a business cost matrix (the cost of funding a
defaulter vs. the opportunity cost of rejecting a good loan). :func:`threshold_sweep`
reports, across candidate thresholds, the confusion-matrix counts and the
expected cost, and identifies the cost-minimizing threshold. This is an
*auxiliary report* — the honest headline stays AUC/PR-AUC/Brier, and NO profit or
ROI figure is claimed.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# quantcore-candidate: new code (threshold/cost sweep); reported only, never
# in the headline.


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """A 2x2 misclassification cost matrix (relative units, not currency).

    Attributes
    ----------
    cost_fn:
        Cost of a false negative (funding a loan that defaults).
    cost_fp:
        Cost of a false positive (rejecting a loan that would have repaid).
    cost_tp:
        Cost (often a benefit, i.e. negative) of a true positive.
    cost_tn:
        Cost of a true negative.
    """

    cost_fn: float = 1.0
    cost_fp: float = 0.2
    cost_tp: float = 0.0
    cost_tn: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this cost matrix."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Immutable result of a cost-matrix threshold sweep.

    Attributes
    ----------
    thresholds:
        The candidate decision thresholds evaluated.
    expected_cost:
        Expected cost per threshold (aligned to ``thresholds``).
    best_threshold:
        The cost-minimizing threshold.
    best_cost:
        The expected cost at ``best_threshold``.
    base_rate:
        The positive (default) base rate of the evaluation set.
    """

    thresholds: list[float]
    expected_cost: list[float]
    best_threshold: float
    best_cost: float
    base_rate: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the sweep."""
        out = asdict(self)
        out["best_threshold"] = float(self.best_threshold)
        out["best_cost"] = float(self.best_cost)
        out["base_rate"] = float(self.base_rate)
        return out


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    cost_matrix: CostMatrix | None = None,
    n_thresholds: int = 101,
) -> ThresholdSweep:
    """Sweep decision thresholds and report the cost-minimizing one.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.
    cost_matrix:
        The misclassification costs; defaults to :class:`CostMatrix`.
    n_thresholds:
        Number of evenly spaced thresholds in ``[0, 1]`` to evaluate.

    Returns
    -------
    ThresholdSweep
        Per-threshold expected cost and the cost-minimizing threshold.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("threshold_sweep is not yet implemented.")
